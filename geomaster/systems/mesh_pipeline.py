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
from geomaster.utils.camera_utils import load_json
import cv2


def save_depth_as_image(depth_tensor, output_dir, prefix, suffix='.png', min_depth=0.0, max_depth=10.0):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch_size, height, width = depth_tensor.shape
    
    min_depth = torch.min(depth_tensor)
    max_depth = torch.max(depth_tensor)
    print(min_depth)
    print(max_depth)
    depth_tensor = torch.clamp(depth_tensor, min_depth, max_depth)
    depth_normalized = (depth_tensor - min_depth) / (max_depth - min_depth)
    
    depth_image = (depth_normalized * 255).byte()
    
    for i in range(batch_size):
        single_depth_image = depth_image[i]
    
        depth_image_pil = Image.fromarray(single_depth_image.cpu().numpy())
        
        filename = os.path.join(output_dir, f"{prefix}_depth_{i}{suffix}")
        depth_image_pil.save(filename)
        print(f"Saved depth image {i} to {filename}")
    
    
def normalize_mesh(mesh, max_lim=1.0):
    center = mesh.vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(mesh.vertices - center, axis=1))
    scale = scale / max_lim
    mesh.vertices = (mesh.vertices - center) / scale
    return mesh, scale, center
    

def prepare_data(source_path, resolution=None):
    dataset_config = { "name":"colmap", "source_path": source_path, 
                        "images":"images", "masks": 'mask', 
                        "resolution":-1, 
                        "data_device":"cuda", "w_mask": True}
    dataset = datasets.make(dataset_config)
    dataset.all_cameras = [_camera.downsample_scale(resolution) for _camera in dataset.all_cameras[::3]]
    cameras = dataset.all_cameras
#    import glob
#    json_path = glob.glob(os.path.join(source_path, '*.json'))
#    print(f"json_path:{json_path[0]}")
#    cameras = load_json(json_path[0])
#    print(f"all_cameras:{cameras}")
#    print(f"images:{camera.image for camera in cameras}")
    imgs = torch.stack([camera.image for camera in cameras], dim=0).cuda()
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).cuda()
    imgs = imgs.permute(0, 3, 1, 2)
    grayimgs = (imgs * weights.view(1, 3, 1, 1)).sum(dim=1)
    try:
        masks = torch.stack([camera.mask for camera in cameras], dim=0).cuda().float() / 255
        # masks = torch.nn.functional.interpolate(masks.unsqueeze(1), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(1)
    except:
        masks = torch.ones_like(grayimgs).cuda().float()
    
    # Load edge images
    edges = []
    for camera in cameras:
        edge_path = str(camera.image_path).replace('images', 'edge').rsplit('.', 1)[0] + '.png'
        if os.path.exists(edge_path):
            edge_image = Image.open(edge_path).convert('L')  # Load as grayscale
            edge_image = np.array(edge_image) / 255.0  # Normalize to [0, 1]
            edge_image = torch.tensor(edge_image, dtype=torch.float32).cuda()
            edge_image = edge_image.unsqueeze(0)  # Add channel dimension
        else:
            print(f"Warning: Edge image not found for {camera.image_path}")
            edge_image = torch.ones_like(grayimgs[0]).cuda()  # Default to all 1s if edge image is missing
        edges.append(edge_image)
    edges = torch.stack(edges, dim=0).cuda()
        
    # load normal
    normals = []
    for camera in cameras:
        # normal_path = str(camera.image_path).replace('images', 'normals')[:-4]+ '.png'
        normal_path = str(camera.image_path).replace('images', 'normals').rsplit('.', 1)[0] + '.png'
        if os.path.exists(normal_path):
            _normal = Image.open(normal_path)
            # _normal = _normal.resize((int(_normal.width * 0.5), int(_normal.height * 0.5)))
            _normal = torch.tensor(np.array(_normal)).cuda().float() / 255 * 2 - 1
            _normal *= -1
            _normal = camera.normal2worldnormal(_normal.cpu())
            
            _normal_norm = torch.norm(_normal, dim=2, keepdim=True)
            _normal_mask = ~((_normal_norm > 1.1) | (_normal_norm < 0.9))
            _normal = _normal / _normal_norm    
        else:
            print('Warning: cannot find gt normals')
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
    return imgs, normals, grayimgs, masks, edges, w2cs, projs, poses, len(imgs)
    
@click.command()
@click.option('--source_path', '-s', type=str, help='Path to dataset')
@click.option('--model_path', '-m', type=str, help='Path to model')
@click.option('--output_path', '-o', type=str, help='Path to model')
@click.option('--num_points', default=30000, type=int, help='Number of points')
@click.option('--num_sample', default=0, type=int, help='Number of samples')
@click.option('--h_patch_size', default=5, type=int, help='Patch size')
@click.option('--ncc_thresh', default=0.05, type=float, help='NCC threshold')
@click.option('--lr', default=0.1, type=float, help='Learning rate')
@click.option('--ncc_weight', default=0.5, type=float, help='NCC weight')
@click.option('--normal_weight', default=0.15, type=float, help='NCC weight')
@click.option('--mask_weight', default=0.0, type=float, help='Mask weight')
@click.option('--atol', default=0.1, type=float, help='Tolerance level for alignment')
@click.option('--resolution', '-r', default=1, type=int, help='Resolution')
@click.option('--save_mid', default=0, type=int, help='Save the intermediate results')
@click.option('--start_edge_len', '-sel', default=0.1, type=float, help='edge_len_lims of MeshOptimizer')
@click.option('--end_edge_len', '-eel', default=0.01, type=float, help='edge_len_lims of MeshOptimizer')
def main(source_path, model_path, output_path, num_points, num_sample, h_patch_size, ncc_thresh, lr, ncc_weight, normal_weight, mask_weight, atol, resolution, save_mid, start_edge_len, end_edge_len): 
    if model_path is None:
        model_path = os.path.join(source_path, 'visual_hull.ply') 
    if output_path is None:
        output_path = model_path[:-4]+f'.v181_{start_edge_len}_{ncc_thresh}_{atol}_{ncc_weight}_{normal_weight}_refined.ply'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(model_path)[:-4]+f'.v181_{start_edge_len}_{ncc_thresh}_{atol}_{ncc_weight}_{normal_weight}_refined.ply')
    num_pixels = (h_patch_size*2+1)**2

    # Load sparse
    imgs, gt_normals, grayimgs, masks, edges, w2cs, projs, poses, num = prepare_data(source_path, resolution)

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
    # vertices, faces = vertices.cuda(), faces.cuda()
    
    mesh_norm = trimesh.load(model_path, process=False, maintain_order=True)
    # mesh_norm, scale_norm, center_norm = normalize_mesh(mesh_norm, max_lim=1)
    vertices = torch.from_numpy(np.array(mesh_norm.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh_norm.faces)).long()
    vertices, faces = vertices.cuda(), faces.cuda()

    inputs_optimizer = MeshOptimizer(vertices.detach(), faces.detach(), ramp=5, edge_len_lims=(end_edge_len, start_edge_len), 
                                     local_edgelen=False) #, laplacian_weight=0.2
    vertices = inputs_optimizer.vertices
    optim_epoch = 200
    # print(f"optim_epoch:{optim_epoch}")
    batch_size = 8
    pbar = tqdm(range(optim_epoch))
    # torch.cuda.empty_cache() 
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
#            b = gt_normal.shape[0]
#            for i in range(b):
#                gt_normals_cpu = gt_normal[i].detach().cpu().numpy()
#                gt_normals_rgb = (gt_normals_cpu + 1.0) / 2.0 * 255
#                gt_normals_rgb = gt_normals_rgb.astype(np.uint8)
#                gt_normals_path = os.path.join(f"gt_normals_{i}.png")
#                gt_normals_rgb_image = Image.fromarray(gt_normals_rgb)
#                gt_normals_rgb_image.save(gt_normals_path)
#            return
            # SAP generation
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            # vertsw = torch.cat([normalized_vertices, torch.ones_like(normalized_vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:,:,:3], faces.long())
            # normals = get_normals(vertsw[:,:,:3], normalized_faces.long())

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
            
            mid_dir = f"{source_path}/mid"
            if not os.path.exists(mid_dir):
                os.makedirs(mid_dir)
            epoch_dir = os.path.join(mid_dir, f"epoch_{iteration}")
            
            if save_mid:
                if iteration % 99 == 0:
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    b = pred_normals.shape[0]
                    print(pred_normals.shape)
                    for i in range(b):
                        
                        pred_normals_cpu = pred_normals[i].detach().cpu().numpy()
                        pred_normals_rgb = (pred_normals_cpu + 1.0) / 2.0 * 255
                        pred_normals_rgb = pred_normals_rgb.astype(np.uint8)
                        pred_normals_path = os.path.join(epoch_dir, f"pred_normals_{i}.png")
                        pred_normals_rgb_image = Image.fromarray(pred_normals_rgb)
                        pred_normals_rgb_image.save(pred_normals_path)
        
                        gt_normals_cpu = gt_normal[i].detach().cpu().numpy()
                        gt_normals_rgb = (gt_normals_cpu + 1.0) / 2.0 * 255
                        gt_normals_rgb = gt_normals_rgb.astype(np.uint8)
                        gt_normals_path = os.path.join(epoch_dir, f"gt_normals_{i}.png")
                        gt_normals_rgb_image = Image.fromarray(gt_normals_rgb)
                        gt_normals_rgb_image.save(gt_normals_path)
            
                        print(f"Saved pred_normals_{i}.png and gt_normals_{i}.png in {epoch_dir}")
    
                    # return
            
            # Compute Mask Loss
            mask_loss = mask_weight * F.mse_loss(pred_mask, mask)
            
            # Compute Normal Loss
            # Create the mask to identify valid pixels
            gt_normal_mask = (gt_normal[..., 3] > 0) & (ref_mask[0] > 0)
            
#            num_zeros = torch.sum(gt_normal_mask == 0).item()
#            num_ones = torch.sum(gt_normal_mask != 0).item()           
#            print(gt_normal_mask.shape)
#            print(f"Number of 0s: {num_zeros}")
#            print(f"Number of 1s: {num_ones}")
#            non_zero_values = gt_normal_mask[gt_normal_mask != 0]
#            print("Non-zero values in src_mask:")
#            print(non_zero_values)
            
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
                
                print('valid_normal_error.numel() = 0')
                return
                num_zeros = torch.sum(rast_out[0, :, :, 3] == 0).item()
                num_ones = torch.sum(rast_out[0, :, :, 3] != 0).item()
                print(rast_out[0, :, :, 3].shape)
                print(f"Number of 0s: {num_zeros}")
                print(f"Number of 1s: {num_ones}")
                non_zero_values = rast_out[0, :, :, 3][rast_out[0, :, :, 3] != 0]
                print("Non-zero values in src_mask:")
                print(non_zero_values)
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
            
            # Combine with edge mask (only compute loss where edge image is 0)
            edge_mask = edges[perm[k:k+1], 0] == 0  # Edge image mask
            edge_mask = edge_mask & (rast_out[0, :, :, 3] > 0)  # Only valid pixels
            
            # Crop edge_mask to match the size of gradient masks
            edge_mask_x = edge_mask[:, :, :-1]
            edge_mask_y = edge_mask[:, :-1, :]
            
            grad_mask_x = grad_mask_x & edge_mask_x
            grad_mask_y = grad_mask_y & edge_mask_y

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

            total_loss = (ncc_loss + mask_loss + normal_loss + 10.0*normal_grad_loss)/ batch_size
            # total_loss = (mask_loss + normal_loss + 0.1*normal_grad_loss)/ batch_size
            mean_ncc_loss += ncc_loss.item() 
            # Optimizer step
            total_loss.backward()
            # torch.cuda.empty_cache()

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
                # np_vertices, np_faces = vertices.detach().cpu().numpy() * scale_norm + center_norm, faces.detach().cpu().numpy()
                save_mesh = trimesh.Trimesh(np_vertices, np_faces, process=False, maintain_order=True)
                # save_mesh = clean_mesh(save_mesh, thresh=0.01)
                save_mesh.export(output_path)

def update_pbar_description(pbar, ncc_loss, mask_loss, sparse_loss):
    des = f'ncc:{ncc_loss.item():.4f} m:{mask_loss.item():.4f} normal:{sparse_loss.item():.4f}'
    # des = f'm:{mask_loss.item():.4f} normal:{sparse_loss.item():.4f}'
    pbar.set_description(des)

if __name__ == '__main__':
    main()