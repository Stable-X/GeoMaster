# code adapted from FastHuman https://github.com/l1346792580123/FastHuman
import os
from os.path import join
from tqdm import tqdm
import trimesh
import torch
import torch.nn.functional as F
from torch.optim import Adam
import nvdiffrast.torch as dr
from geomaster.utils.ncc_utils import build_patch_offset, NCC
from geomaster.utils.mesh_utils import get_normals
from geomaster.models.mesh import MeshOptimizer, gen_inputs, clean_mesh
import click
from gaustudio import datasets
from PIL import Image
import numpy as np
import warnings


def prepare_data(source_path, resolution=None):
    dataset_config = { "name":"colmap", "source_path": source_path, 
                        "images":"images", "masks": 'mask', 
                        "resolution":-1, 
                        "data_device":"cuda", "w_mask": True}
    dataset = datasets.make(dataset_config)
    dataset.all_cameras = [_camera.downsample_scale(resolution) for _camera in dataset.all_cameras[::3]]
    cameras = dataset.all_cameras
    
    imgs = torch.stack([camera.image for camera in cameras], dim=0).cuda()
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).cuda()
    imgs = imgs.permute(0, 3, 1, 2)
    grayimgs = (imgs * weights.view(1, 3, 1, 1)).sum(dim=1)
    try:
        masks = torch.stack([camera.mask for camera in cameras], dim=0).cuda().float() / 255
    except:
        masks = torch.ones_like(grayimgs).cuda().float()
        
    # load normal
    normals = []
    for camera in cameras:
        normal_path = str(camera.image_path).replace('images', 'normals').rsplit('.', 1)[0] + '.png'
        if os.path.exists(normal_path):
            _normal = Image.open(normal_path)
            _normal = torch.tensor(np.array(_normal)).cuda().float() / 255 * 2 - 1
            _normal *= -1
            _normal = camera.normal2worldnormal(_normal.cpu())
            
            _normal_norm = torch.norm(_normal, dim=2, keepdim=True)
            _normal_mask = ~((_normal_norm > 1.1) | (_normal_norm < 0.9))
            _normal = _normal / _normal_norm    
        else:
            warnings.warn('Warning: cannot find gt normals')
            _normal = torch.zeros_like(imgs[0]).cuda().permute(1, 2, 0)
            _normal_mask = torch.zeros_like(imgs[0][0:1]).cuda().permute(1, 2, 0)
        _normal = torch.cat([_normal, _normal_mask], dim=2)
        _normal = torch.nn.functional.interpolate(_normal.permute(2, 0, 1).unsqueeze(0), size=(camera.image_height, camera.image_width),
                                                  mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
        normals.append(_normal)
    normals = torch.stack(normals, dim=0).cuda().contiguous()

    w2cs = torch.stack([camera.extrinsics.T for camera in cameras], dim=0).cuda()
    projs = torch.stack([camera.projection_matrix for camera in cameras], dim=0).cuda()
    poses = w2cs.permute(0, 2, 1).contiguous()
    return imgs, normals, grayimgs, masks, w2cs, projs, poses, len(imgs)
    
@click.command()
@click.option('--source_path', '-s', type=str, help='Path to dataset')
@click.option('--model_path', '-m', type=str, help='Path to model')
@click.option('--output_path', '-o', type=str, help='Path to model')
@click.option('--num_points', default=30000, type=int, help='Number of points')
@click.option('--num_sample', default=0, type=int, help='Number of samples')
@click.option('--h_patch_size', default=5, type=int, help='Patch size')
@click.option('--ncc_thresh', default=0.5, type=float, help='NCC threshold')
@click.option('--lr', default=0.05, type=float, help='Learning rate')
@click.option('--ncc_weight', default=0.15, type=float, help='NCC weight')
@click.option('--normal_weight', default=0.5, type=float, help='NCC weight')
@click.option('--mask_weight', default=0., type=float, help='Mask weight')
@click.option('--atol', default=0.01, type=float, help='Tolerance level for alignment')
@click.option('--start_edge_len', '-sel', default=0.1, type=float, help='edge_len_lims of MeshOptimizer')
@click.option('--end_edge_len', '-eel', default=0.01, type=float, help='edge_len_lims of MeshOptimizer')
def main(source_path, model_path, output_path, num_points, num_sample, h_patch_size, ncc_thresh, lr, ncc_weight, normal_weight, mask_weight, atol, resolution, start_edge_len, end_edge_len):
    if model_path is None:
        model_path = os.path.join(source_path, 'visual_hull.ply')
    if output_path is None:
        output_path = model_path[:-4]+'.refined.ply'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(model_path)[:-4]+'.refined.ply')
    num_pixels = (h_patch_size*2+1)**2

    # Load sparse
    imgs, gt_normals, grayimgs, masks, w2cs, projs, poses, num = prepare_data(source_path, resolution)
    _, _, image_height, image_width = imgs.shape
    resolution = (image_height, image_width)

    pairs = []
    intervals = [-2, -1, 1, 2]
    for randidx in range(num):
        pairs.append(torch.tensor([randidx+itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < num))]).cuda())
    offsets = build_patch_offset(h_patch_size, pairs[0].device).float()
    
    # Generate input mesh
    glctx = dr.RasterizeGLContext()
    vertices, faces = gen_inputs(model_path, num_sample)
    vertices, faces = vertices.cuda(), faces.cuda()
    inputs_optimizer = MeshOptimizer(vertices.detach(), faces.detach(), ramp=5, edge_len_lims=(end_edge_len, start_edge_len), 
                                     local_edgelen=False)
    vertices = inputs_optimizer.vertices
    
    optim_epoch = 100
    batch_size = 8
    pbar = tqdm(range(optim_epoch))

    # Main optimization loop
    for iteration in pbar:
        perm = torch.randperm(num).cuda()
        mean_ncc_loss = 0
        for k in range(0, batch_size):
            ref_w2c = w2cs[perm[k:k+1]]
            ref_proj = projs[perm[k:k+1]]
            ref_gray = grayimgs[perm[k:k+1]]
            ref_normal = gt_normals[perm[k:k+1]]
            ref_mask = masks[perm[k:k+1]]
            src_w2c = w2cs[pairs[perm[k]]]
            src_pose = poses[pairs[perm[k]]]
            src_proj = projs[pairs[perm[k]]]
            src_gray = grayimgs[pairs[perm[k]]]
            src_normal = gt_normals[pairs[perm[k]]]
            src_mask = masks[pairs[perm[k]]]

            w2c = torch.cat([ref_w2c, src_w2c])
            proj = torch.cat([ref_proj, src_proj])
            gt_normal = torch.cat([ref_normal, src_normal])
            mask = torch.cat([ref_mask, src_mask])
            n = w2c.shape[0]

            # SAP generation
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())

            int32_faces = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(glctx, proj_verts, int32_faces, resolution=resolution)
            
            # render depth
            feat = torch.cat([rot_verts[:,:,:3], torch.ones_like(vertsw[:,:,:1]), vertsw[:,:,:3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, int32_faces)
            rast_verts = feat[:,:,:,:3].contiguous()
            pred_mask = feat[:,:,:,3:4].contiguous()
            rast_points = feat[:,:,:,4:7].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, int32_faces).squeeze(-1)

            # render normal
            feat, _ = dr.interpolate(normals, rast_out, int32_faces)
            pred_normals = feat.contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, int32_faces)
            pred_normals = F.normalize(pred_normals,p=2,dim=3)
            
            # Compute Mask Loss
            mask_loss = mask_weight * F.mse_loss(pred_mask, mask)
            
            # Compute Normal Loss
            # Create the mask to identify valid pixels
            gt_normal_mask = (gt_normal[..., 3] > 0) & (ref_mask[0] > 0)
            gt_normal_mask = gt_normal_mask & (rast_out[0, :, :, 3] > 0)

            # Compute the normal error
            normal_error = (1 - (pred_normals * gt_normal[..., :3]).sum(dim=3))

            # Filter the normal error using the gt_normal_mask
            valid_normal_error = normal_error[gt_normal_mask]

            # Ignore NaN values in the computation of the mean
            valid_normal_error = valid_normal_error[~torch.isnan(valid_normal_error)]

            # Calculate the mean of the valid normal errors
            if valid_normal_error.numel() > 0:
                normal_loss = normal_weight * valid_normal_error.mean()
            else:
                warnings.warn('Warning: normal_loss is None')
                normal_loss = torch.tensor(0.0, device=pred_normals.device)  # or any appropriate default value or handling
            
            # Compute gradients for predicted normals
            pred_grad_x = pred_normals[:, :, 1:, :] - pred_normals[:, :, :-1, :]
            pred_grad_y = pred_normals[:, 1:, :, :] - pred_normals[:, :-1, :, :]
            
            # Compute gradients for ground truth normals
            gt_grad_x = gt_normal[:, :, 1:, :3] - gt_normal[:, :, :-1, :3]
            gt_grad_y = gt_normal[:, 1:, :, :3] - gt_normal[:, :-1, :, :3]

            # Create gradient masks
            grad_mask_x = gt_normal_mask[:, :, 1:] & gt_normal_mask[:, :, :-1]
            grad_mask_y = gt_normal_mask[:, 1:, :] & gt_normal_mask[:, :-1, :]

            # Compute gradient errors
            grad_error_x = F.mse_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x], reduction='mean')
            grad_error_y = F.mse_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y], reduction='mean')

            # Compute normal gradient loss
            normal_grad_loss = normal_weight * (grad_error_x + grad_error_y) / 2
                
            # Compute NCC Loss
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

            sampled_ref_gray = ref_gray[:, vv, uu].reshape(1, npoints, num_pixels)
            sampled_src_gray = F.grid_sample(src_gray.unsqueeze(1), grid.view(n-1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_gray = sampled_src_gray.reshape(n-1, npoints, num_pixels)
            ncc_values = NCC(sampled_ref_gray, sampled_src_gray, ref_valid_mask, src_valid_mask) # nview npoints

            ncc_mask = (ncc_values > ncc_thresh) & (src_valid_mask.sum(2) > num_pixels*0.75)

            if not (ncc_values[ncc_mask]<1).all():
                ncc_loss = torch.tensor(0)
            else:
                ncc_values = torch.clamp(ncc_values,max=1.0)
                ncc_loss = ncc_weight * torch.sum((torch.ones_like(ncc_values)-ncc_values)*ncc_mask) / ncc_mask.sum()

            total_loss = (ncc_loss + mask_loss + normal_loss + normal_grad_loss)/ batch_size
            mean_ncc_loss += ncc_loss.item() 
            # Optimizer step
            total_loss.backward()

        inputs_optimizer.step()
        inputs_optimizer.zero_grad()

        # Update progress bar description
        update_pbar_description(pbar, ncc_loss, mask_loss, normal_loss)
        mean_ncc_loss = 0
        vertices, faces = inputs_optimizer.remesh()
        
        if iteration % 10 == 0:
            # Save intermediate results
            with torch.no_grad():
                np_vertices, np_faces = vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
                save_mesh = trimesh.Trimesh(np_vertices, np_faces, process=False, maintain_order=True)
                # save_mesh = clean_mesh(save_mesh, thresh=0.01)
                save_mesh.export(output_path)

def update_pbar_description(pbar, ncc_loss, mask_loss, sparse_loss):
    des = f'ncc:{ncc_loss.item():.4f} m:{mask_loss.item():.4f} normal:{sparse_loss.item():.4f}'
    pbar.set_description(des)

if __name__ == '__main__':
    main()