# code adapted from FastHuman https://github.com/l1346792580123/FastHuman
import os
from os.path import join
from tqdm import tqdm
import trimesh
import torch
import torch.nn.functional as F
from torch.optim import Adam
import nvdiffrast.torch as dr
from geomaster.systems.ncc_utils import build_patch_offset, NCC, SSIM
from geomaster.models.sap import PSR2Mesh, DPSR, sap_generate, gen_inputs
import click
from gaustudio import datasets

def prepare_data(source_path, resolution=None):
    dataset_config = { "name":"colmap", "source_path": source_path, 
                        "images":"images", "masks": 'mask', 
                        "resolution":-1, 
                        "data_device":"cuda", "w_mask": True}
    dataset = datasets.make(dataset_config)
    cameras = dataset.all_cameras
    
    imgs = torch.stack([camera.image for camera in cameras], dim=0).cuda()
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).cuda()
    imgs = imgs.permute(0, 3, 1, 2)
    grayimgs = (imgs * weights.view(1, 3, 1, 1)).sum(dim=1)
    try:
        masks = torch.stack([camera.mask for camera in cameras], dim=0).cuda().float() / 255
    except:
        masks = torch.ones_like(grayimgs).cuda().float()
    w2cs = torch.stack([camera.extrinsics.T for camera in cameras], dim=0).cuda()
    projs = torch.stack([camera.projection_matrix for camera in cameras], dim=0).cuda()
    poses = w2cs.permute(0, 2, 1).contiguous()
    return imgs[::3], grayimgs[::3], masks[::3], w2cs[::3], projs[::3], poses[::3], len(imgs[::3])
    
@click.command()
@click.option('--source_path', '-s', type=str, help='Path to dataset')
@click.option('--model_path', '-m', type=str, help='Path to model')
@click.option('--output_path', '-o', type=str, help='Path to model')
@click.option('--sap_res', default=256, type=int, help='SAP resolution')
@click.option('--sig', default=2, type=int, help='Sigma value')
@click.option('--num_points', default=30000, type=int, help='Number of points')
@click.option('--num_sample', default=50000, type=int, help='Number of samples')
@click.option('--h_patch_size', default=5, type=int, help='Patch size')
@click.option('--ncc_thresh', default=0.5, type=float, help='NCC threshold')
@click.option('--lr', default=0.001, type=float, help='Learning rate')
@click.option('--rgb_ncc', default=False, type=bool, help='RGB NCC flag')
@click.option('--use_sparse', default=False, type=bool, help='Use sparse flag')
@click.option('--ncc_weight', default=5, type=float, help='NCC weight')
@click.option('--mask_weight', default=20, type=float, help='Mask weight')
@click.option('--atol', default=0.01, type=float, help='Tolerance level for alignment')
def main(source_path, model_path, output_path, sap_res, sig, num_points, num_sample, h_patch_size, ncc_thresh, lr, rgb_ncc, use_sparse, ncc_weight, mask_weight, atol):
    if output_path is None:
        output_path = model_path[:-4]+'.refined.ply'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(model_path)[:-4]+'.refined.ply')
    num_pixels = (h_patch_size*2+1)**2

    # Load sparse
    imgs, grayimgs, masks, w2cs, projs, poses, num = prepare_data(source_path)
    _, _, image_height, image_width = imgs.shape
    resolution = (image_height, image_width)

    pairs = []
    intervals = [-2, -1, 1, 2]
    for randidx in range(num):
        pairs.append(torch.tensor([randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < num))]).cuda())
    offsets = build_patch_offset(h_patch_size, pairs[0].device).float()
    
    # Initialize SAP and context
    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((sap_res, sap_res, sap_res), sig).cuda()
    glctx = dr.RasterizeGLContext()
    
    # Generate input mesh
    inputs, center, scale = gen_inputs(model_path, num_sample)
    inputs, center, scale = inputs.cuda(), center.cuda(), scale.cuda()

    inputs.requires_grad_(True)
    inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])
    optim_epoch = 10
    pbar = tqdm(range(optim_epoch))

    # Main optimization loop
    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num):
            ref_w2c = w2cs[perm[k:k+1]]
            ref_proj = projs[perm[k:k+1]]
            ref_gray = grayimgs[perm[k:k+1]]
            ref_img = imgs[perm[k]]
            ref_mask = masks[perm[k:k+1]]
            src_w2c = w2cs[pairs[perm[k]]]
            src_pose = poses[pairs[perm[k]]]
            src_proj = projs[pairs[perm[k]]]
            src_gray = grayimgs[pairs[perm[k]]]
            src_img = imgs[pairs[perm[k]]]
            src_mask = masks[pairs[perm[k]]]

            w2c = torch.cat([ref_w2c, src_w2c])
            proj = torch.cat([ref_proj, src_proj])
            mask = torch.cat([ref_mask, src_mask])
            n = w2c.shape[0]

            # SAP generation
            vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
            feat = torch.cat([rot_verts[:,:,:3], torch.ones_like(vertsw[:,:,:1]), vertsw[:,:,:3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, faces)
            rast_verts = feat[:,:,:,:3].contiguous()
            pred_mask = feat[:,:,:,3:4].contiguous()
            rast_points = feat[:,:,:,4:7].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, faces).squeeze(-1)

            # Compute NCC loss
            valid_mask = (rast_out[0,:,:,3] > 0) & (ref_mask[0] > 0)
            ref_valid_idx = torch.where(valid_mask)
            rand_idx = torch.randperm(len(ref_valid_idx[0]))
            ref_idx = [item[rand_idx][:num_points] for item in ref_valid_idx] # part sample
            uv = torch.stack([ref_idx[1], ref_idx[0]], dim=1).unsqueeze(1) # npoints 1 2
            npoints = uv.shape[0]
            pixels = (uv + offsets).reshape(-1,2) # npoints*npixels 2
            uu = torch.clamp(pixels[:,0], 0, image_width-1).long()
            vv = torch.clamp(pixels[:,1], 0, image_height-1).long()
            uv_mask = ((pixels[:,0] >= 0) & (pixels[:,0] < image_width) 
                       & (pixels[:,1] >= 0) & (pixels[:,1] < image_height)).reshape(1, npoints, num_pixels)
            ref_points = rast_points[0][vv,uu]
            ref_valid_mask = valid_mask[vv,uu].reshape(1, npoints, num_pixels) & uv_mask

            src_verts = (src_pose[:,:3,:3]@ref_points.permute(1,0).contiguous() + src_pose[:,:3,3:4]).permute(0,2,1).contiguous()
            src_depth = src_verts[:,:,2].reshape(n-1, npoints, num_pixels)
            src_f = torch.stack([src_proj[:,0,0], src_proj[:,1,1]], dim=1).unsqueeze(1)
            src_c = torch.stack([src_proj[:,2,0], src_proj[:,2,1]], dim=1).unsqueeze(1)
            grid = (src_verts[:,:,:2] / (src_verts[:,:,2:3]+1e-8)) * src_f + src_c

            sampled_src_depth = F.grid_sample(rast_verts[1:,:,:,2:3].permute(0,3,1,2).contiguous(), grid.view(n-1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_depth = sampled_src_depth.reshape(n-1, npoints, num_pixels)

            sampled_src_mask = F.grid_sample(src_mask.unsqueeze(-1).permute(0,3,1,2).contiguous(), grid.view(n-1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_mask = sampled_src_mask.reshape(n-1, npoints, num_pixels)

            src_valid_mask = ref_valid_mask & torch.isclose(sampled_src_depth, src_depth, atol=atol) & (sampled_src_mask>0)

            if rgb_ncc:
                sampled_ref_img = ref_img[:, vv,uu].reshape(npoints, num_pixels, 3).permute(2,0,1).contiguous()
                sampled_src_img = F.grid_sample(src_img.contiguous(), grid.view(n-1, -1, 1, 2), align_corners=False).squeeze()
                sampled_src_img = sampled_src_img.reshape(n-1, 3, npoints, num_pixels)
                ncc_values = 0
                for j in range(3):
                    ncc_values = ncc_values + SSIM(sampled_ref_img[j:j+1], sampled_src_img[:,j], ref_valid_mask, src_valid_mask)
                ncc_values = ncc_values / 3
            else:
                sampled_ref_gray = ref_gray[:, vv, uu].reshape(1, npoints, num_pixels)
                sampled_src_gray = F.grid_sample(src_gray.unsqueeze(1), grid.view(n-1, -1, 1, 2), align_corners=False).squeeze()
                sampled_src_gray = sampled_src_gray.reshape(n-1, npoints, num_pixels)
                ncc_values = SSIM(sampled_ref_gray, sampled_src_gray, ref_valid_mask, src_valid_mask) # nview npoints

            ncc_mask = (ncc_values > ncc_thresh) & (src_valid_mask.sum(2) > num_pixels*0.75)

            # assert (ncc_values[ncc_mask]<1).all()
            ncc_values = torch.clamp(ncc_values,max=1.0)


            ncc_loss = ncc_weight * torch.sum((torch.ones_like(ncc_values)-ncc_values)*ncc_mask) / ncc_mask.sum()
            # Compute mask loss
            mask_loss = torch.zeros_like(ncc_loss)

            # Compute sparse point loss if applicable
            sparse_loss = torch.zeros_like(ncc_loss)
            
            total_loss = ncc_loss + mask_loss + sparse_loss

            # Optimizer step
            inputs_optimizer.zero_grad()
            total_loss.backward()
            inputs_optimizer.step()

            # Update progress bar description
            update_pbar_description(pbar, ncc_loss, mask_loss, sparse_loss)

        # Save intermediate results
        with torch.no_grad():
            vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)

            save_verts = vertices.squeeze(0).detach().cpu().numpy()
            np_faces = faces.squeeze(0).detach().cpu().long().numpy()
            save_mesh = trimesh.Trimesh(save_verts, np_faces, process=False, maintain_order=True)
            
            save_mesh.export(output_path)

            inputs, center, scale = gen_inputs(output_path, num_sample)
            inputs = inputs.cuda()
            inputs.requires_grad_(True)
            center = center.cuda()
            scale = scale.cuda()

            del inputs_optimizer
            inputs_optimizer = Adam([{'params': inputs, 'lr': lr}])

def update_pbar_description(pbar, ncc_loss, mask_loss, sparse_loss):
    des = f'ncc:{ncc_loss.item():.4f} m:{mask_loss.item():.4f} sp:{sparse_loss.item():.4f}'
    pbar.set_description(des)

if __name__ == '__main__':
    main()