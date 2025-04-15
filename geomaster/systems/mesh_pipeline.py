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
from geomaster.utils.depth_utils import read_depth_meter
from geomaster.models.mesh import MeshOptimizer, gen_inputs, clean_mesh
import click
from gaustudio import datasets
from PIL import Image
import numpy as np
from geomaster.utils.camera_utils import load_json
import cv2


DEFAULT_PARAMS = {
    'object': {
        'num_points': 30000,
        'num_sample': 0,
        'h_patch_size': 20,
        'ncc_thresh': 0.05,
        'lr': 0.1,
        'ncc_weight': 0.5,
        'normal_weight': 0.0,
        'normal_grad_weight': 0.5,
        'mask_weight': 0.5,
        'atol': 0.1,
        'resolution': 1,
        'save_mid': 0,
        'start_edge_len': 0.1,
        'end_edge_len': 0.01,
        'laplacian_weight': 0.02
    },
    'scene': {
        'num_points': 30000,
        'num_sample': 0,
        'h_patch_size': 5,
        'ncc_thresh': 0.5,
        'lr': 0.05,
        'ncc_weight': 0.15,
        'normal_weight': 0.2,
        'normal_l1_weight': 0.1,
        'depth_weight': 0.8,
        'mask_weight': 0.0,
        'atol': 0.01,
        'resolution': 1,
        'start_edge_len': 0.04,
        'end_edge_len': 0.001,
        'laplacian_weight': 0.02
    }
}


def merge_params(data_type, **kwargs):
    if data_type not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown data_type: {data_type}")
    params = DEFAULT_PARAMS[data_type].copy()
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    return params


def prepare_data(source_path, resolution=None):
    def load_edge_image(camera, resolution, grayimg):
        edge_path = str(camera.image_path).replace('images', 'edge').rsplit('.', 1)[0] + '.png'
        if os.path.exists(edge_path):
            edge_image = Image.open(edge_path).convert('L')
            edge_image = np.array(edge_image) / 255.0
            edge_tensor = torch.tensor(edge_image, dtype=torch.float32).cuda().unsqueeze(0)  # (1, H, W)
        else:
            print(f"Warning: Edge image not found for {camera.image_path}")
            edge_tensor = torch.ones_like(grayimg).cuda()
        if resolution and resolution > 0:
            edge_tensor = F.interpolate(edge_tensor.unsqueeze(0), scale_factor=1.0 / resolution, mode='bilinear', align_corners=False).squeeze(0)
        return edge_tensor

    dataset_config = {"name": "colmap", "source_path": source_path,
                      "images": "images", "masks": 'mask',
                      "resolution": -1,
                      "data_device": "cuda", "w_mask": True}
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

    try:
        depths = torch.stack(
            [torch.from_numpy(
                read_depth_meter(image_path.replace("/images/", "/mono_depths_aligned/").replace(".jpg", ".png")
                                 )).float().unsqueeze(0)
             for image_path in [str(cam_info.image_path) for cam_info in cameras]],
            dim=0
        ).cuda()
    except Exception as e:
        print("Warning: could not read depth images.\n", e)
        depths = None

    # Load edge images
    try:
        edges = torch.stack(
            [load_edge_image(camera, resolution, grayimgs[0])
             for camera in cameras],
            dim=0
        ).cuda()
    except Exception as e:
        print("Warning: could not read edge images.\n", e)
        edges = None

    # load normal
    normals = []
    for camera in cameras:
        # normal_path = str(camera.image_path).replace('images', 'normals')[:-4]+ '.png'
        normal_path = str(camera.image_path).replace('images', 'normals').rsplit('.', 1)[0] + '.png'
        if os.path.exists(normal_path):
            _normal = Image.open(normal_path)
            # _normal = _normal.resize((int(_normal.width * 0.5), int(_normal.height * 0.5)))

            # # Convert coordinate for normals from gm-prepare-data
            # _normal_array = np.array(_normal)
            # _normal_array = _normal_array[:, :, [1, 0, 2]]
            # _normal_array[:, :, 1] = 255 - _normal_array[:, :, 1]

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
    return imgs, depths, normals, grayimgs, masks, edges, w2cs, projs, poses, len(imgs)


def refine_object(source_path, model_path, output_path, num_points, num_sample, h_patch_size, ncc_thresh, lr, ncc_weight,
                  normal_weight, normal_grad_weight, mask_weight, atol, resolution, save_mid, start_edge_len, end_edge_len):
    if model_path is None:
        model_path = os.path.join(source_path, 'visual_hull.ply')
    if output_path is None:
        output_path = model_path[:-4] + f'.refined.ply'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(model_path)[:-4] + f'.refined.ply')
    num_pixels = (h_patch_size * 2 + 1) ** 2

    # Load sparse
    imgs, _, gt_normals, grayimgs, masks, edges, w2cs, projs, poses, num = prepare_data(source_path, resolution)

    _, _, image_height, image_width = imgs.shape
    resolution = (image_height, image_width)

    pairs = []
    intervals = [-2, -1, 1, 2]
    for randidx in range(num):
        pairs.append(torch.tensor([randidx + itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < num))]).cuda())
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
                                     local_edgelen=False)  # , laplacian_weight=0.2
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
            ref_w2c = w2cs[perm[k:k + 1]]
            ref_proj = projs[perm[k:k + 1]]
            ref_gray = grayimgs[perm[k:k + 1]]
            ref_normal = gt_normals[perm[k:k + 1]]
            ref_mask = masks[perm[k:k + 1]]
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
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:, 0:1])], axis=1).unsqueeze(0).expand(n, -1, -1)
            # vertsw = torch.cat([normalized_vertices, torch.ones_like(normalized_vertices[:,0:1])], axis=1).unsqueeze(0).expand(n,-1,-1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:, :, :3], faces.long())
            # normals = get_normals(vertsw[:,:,:3], normalized_faces.long())

            int32_faces = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(glctx, proj_verts, int32_faces, resolution=resolution)

            # render depth
            feat = torch.cat([rot_verts[:, :, :3], torch.ones_like(vertsw[:, :, :1]), vertsw[:, :, :3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, int32_faces)
            rast_verts = feat[:, :, :, :3].contiguous()
            pred_mask = feat[:, :, :, 3:4].contiguous()
            rast_points = feat[:, :, :, 4:7].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, int32_faces).squeeze(-1)

            # render normal
            feat, _ = dr.interpolate(normals, rast_out, int32_faces)
            pred_normals = feat.contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, int32_faces)
            pred_normals = F.normalize(pred_normals, p=2, dim=3)

            if save_mid:
                mid_dir = f"{source_path}/mid"
                if not os.path.exists(mid_dir):
                    os.makedirs(mid_dir)
                epoch_dir = os.path.join(mid_dir, f"epoch_{iteration}")
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

            gt_normal_mask = gt_normal_mask & (rast_out[0, :, :, 3] > 0)

            # Compute the normal error
            normal_error = (1 - (pred_normals * gt_normal[..., :3]).sum(dim=3))

            # Filter the normal error using the gt_normal_mask
            valid_normal_error = normal_error[gt_normal_mask]

            # Ignore NaN values in the computation of the mean
            valid_normal_error = valid_normal_error[~torch.isnan(valid_normal_error)]

            if len(valid_normal_error) > 0:
                normal_error_threshold = torch.median(valid_normal_error) * 10
            else:
                normal_error_threshold = 1.0
            static_mask = normal_error > normal_error_threshold
            dynamic_mask = ~static_mask & gt_normal_mask
            valid_normal_error = normal_error[dynamic_mask]
            valid_normal_error = valid_normal_error[~torch.isnan(valid_normal_error)]

            # Calculate the mean of the valid normal errors
            if valid_normal_error.numel() > 0:
                normal_loss = normal_weight * valid_normal_error.mean()
            else:
                print('valid_normal_error.numel() = 0')
                normal_loss = torch.tensor(0.0, device=pred_normals.device)  # or any appropriate default value or handling

            # Compute gradients for predicted normals
            pred_grad_x = pred_normals[:, :, 1:, :] - pred_normals[:, :, :-1, :]
            pred_grad_y = pred_normals[:, 1:, :, :] - pred_normals[:, :-1, :, :]

            # Compute gradients for ground truth normals - fix dimension mismatch
            gt_grad_x = gt_normal[:, :, 1:, :3] - gt_normal[:, :, :-1, :3]
            gt_grad_y = gt_normal[:, 1:, :, :3] - gt_normal[:, :-1, :, :3]  # Keep :3 for both tensors

            # Calculate gradient magnitudes
            pred_grad_mag_x = torch.norm(pred_grad_x, dim=3)
            pred_grad_mag_y = torch.norm(pred_grad_y, dim=3)
            gt_grad_mag_x = torch.norm(gt_grad_x, dim=3)
            gt_grad_mag_y = torch.norm(gt_grad_y, dim=3)

            # Calculate adaptive threshold based on predicted gradient statistics
            valid_pred_grads = torch.cat([pred_grad_mag_x[gt_normal_mask[:, :, 1:]],
                                          pred_grad_mag_y[gt_normal_mask[:, 1:, :]]])
            if len(valid_pred_grads) > 0:
                small_grad_threshold = torch.median(valid_pred_grads) * 0.2
            else:
                small_grad_threshold = 0.01  # fallback value

            # Create mask for regions where both predicted and ground truth gradients are small
            small_grad_x = (pred_grad_mag_x < small_grad_threshold) & (gt_grad_mag_x < small_grad_threshold)
            small_grad_y = (pred_grad_mag_y < small_grad_threshold) & (gt_grad_mag_y < small_grad_threshold)

            # Expand small_grad masks to match original image size
            small_grad_mask = torch.zeros_like(static_mask, dtype=torch.bool)
            small_grad_mask[:, :, :-1] |= small_grad_x
            small_grad_mask[:, :-1, :] |= small_grad_y

            # Update static_mask to include areas with small gradients
            static_mask = static_mask | small_grad_mask
            dynamic_mask = ~static_mask & gt_normal_mask

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
            edge_mask = edges[perm[k:k + 1], 0] == 0  # Edge image mask
            edge_mask = edge_mask & (rast_out[0, :, :, 3] > 0)  # Only valid pixels

            # Crop edge_mask to match the size of gradient masks
            edge_mask_x = edge_mask[:, :, :-1]
            edge_mask_y = edge_mask[:, :-1, :]

            # Ensure mask is of type bool for bitwise operations
            mask_x = (mask[:, :, 1:] > 0) & (mask[:, :, :-1] > 0)
            mask_y = (mask[:, 1:, :] > 0) & (mask[:, :-1, :] > 0)

            # Final gradient masks
            grad_mask_x = grad_mask_x & edge_mask_x & mask_x
            grad_mask_y = grad_mask_y & edge_mask_y & mask_y

            # Compute gradient errors
            grad_error_x = F.mse_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x], reduction='mean')
            grad_error_y = F.mse_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y], reduction='mean')

            # Compute normal gradient loss
            normal_grad_loss = normal_grad_weight * (grad_error_x + grad_error_y) / 2

            # Compute NCC Loss
            valid_mask = (rast_out[0, :, :, 3] > 0) & (ref_mask[0] > 0) & dynamic_mask[0]
            ref_valid_idx = torch.where(valid_mask)
            rand_idx = torch.randperm(len(ref_valid_idx[0]))
            ref_idx = [item[rand_idx][:num_points] for item in ref_valid_idx]  # part sample
            uv = torch.stack([ref_idx[1], ref_idx[0]], dim=1).unsqueeze(1)  # npoints 1 2
            npoints = uv.shape[0]
            pixels = (uv + offsets).reshape(-1, 2)  # npoints*npixels 2
            uu = torch.clamp(pixels[:, 0], 0, image_width - 1).long()
            vv = torch.clamp(pixels[:, 1], 0, image_height - 1).long()
            uv_mask = ((pixels[:, 0] >= 0) & (pixels[:, 0] < image_width)
                       & (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height)).reshape(1, npoints, num_pixels)
            ref_points = rast_points[0][vv, uu]
            ref_valid_mask = valid_mask[vv, uu].reshape(1, npoints, num_pixels) & uv_mask

            src_verts = (src_pose[:, :3, :3] @ ref_points.permute(1, 0).contiguous() + src_pose[:, :3, 3:4]).permute(0, 2, 1).contiguous()
            src_depth = src_verts[:, :, 2].reshape(n - 1, npoints, num_pixels)
            src_f = torch.stack([src_proj[:, 0, 0], src_proj[:, 1, 1]], dim=1).unsqueeze(1)
            src_c = torch.stack([src_proj[:, 2, 0], src_proj[:, 2, 1]], dim=1).unsqueeze(1)
            grid = (src_verts[:, :, :2] / (src_verts[:, :, 2:3] + 1e-8)) * src_f + src_c

            sampled_src_depth = F.grid_sample(rast_verts[1:, :, :, 2:3].permute(0, 3, 1, 2).contiguous(), grid.view(n - 1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_depth = sampled_src_depth.reshape(n - 1, npoints, num_pixels)

            sampled_src_mask = F.grid_sample(src_mask.unsqueeze(-1).permute(0, 3, 1, 2).contiguous(), grid.view(n - 1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_mask = sampled_src_mask.reshape(n - 1, npoints, num_pixels)

            src_valid_mask = ref_valid_mask & torch.isclose(sampled_src_depth, src_depth, atol=atol) & (sampled_src_mask > 0)

            sampled_ref_gray = ref_gray[:, vv, uu].reshape(1, npoints, num_pixels)
            sampled_src_gray = F.grid_sample(src_gray.unsqueeze(1), grid.view(n - 1, -1, 1, 2), align_corners=False).squeeze()
            sampled_src_gray = sampled_src_gray.reshape(n - 1, npoints, num_pixels)
            ncc_values = NCC(sampled_ref_gray, sampled_src_gray, ref_valid_mask, src_valid_mask)  # nview npoints

            ncc_mask = (ncc_values > ncc_thresh) & (src_valid_mask.sum(2) > num_pixels * 0.75)

            if not (ncc_values[ncc_mask] < 1).all():
                ncc_loss = torch.tensor(0)
            else:
                ncc_values = torch.clamp(ncc_values, max=1.0)
                ncc_loss = ncc_weight * torch.sum((torch.ones_like(ncc_values) - ncc_values) * ncc_mask) / ncc_mask.sum()

            total_loss = (ncc_loss + mask_loss + normal_loss + 10.0 * normal_grad_loss) / batch_size
            # total_loss = (mask_loss + normal_loss + 0.1*normal_grad_loss)/ batch_size
            mean_ncc_loss += ncc_loss.item()
            # Optimizer step
            total_loss.backward()
            # torch.cuda.empty_cache()

        inputs_optimizer.step()
        inputs_optimizer.zero_grad()

        # Update progress bar description
        update_pbar_description_object(pbar, ncc_loss, mask_loss, normal_grad_loss)
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


def refine_scene(source_path, model_path, output_path, num_points, num_sample, h_patch_size, ncc_thresh, lr,
         ncc_weight, normal_weight, mask_weight, normal_l1_weight, depth_weight, atol, resolution,
         start_edge_len, end_edge_len, laplacian_weight):
    if model_path is None:
        model_path = os.path.join(source_path, 'visual_hull.ply')
    if output_path is None:
        output_path = model_path[:-4] + '.refined.ply'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(model_path)[:-4] + '.refined.ply')
    # num_pixels = (h_patch_size * 2 + 1) ** 2

    # Load sparse
    imgs, depths, gt_normals, grayimgs, masks, _, w2cs, projs, poses, num = prepare_data(source_path, resolution)
    _, _, image_height, image_width = imgs.shape
    resolution = (image_height, image_width)

    pairs = []
    intervals = [-2, -1, 1, 2]
    for randidx in range(num):
        pairs.append(torch.tensor(
            [randidx + itv for itv in intervals if ((itv + randidx > 0) and (itv + randidx < num))]).cuda())
    # offsets = build_patch_offset(h_patch_size, pairs[0].device).float()

    # Generate input mesh
    glctx = dr.RasterizeGLContext()
    vertices, faces = gen_inputs(model_path, num_sample)
    vertices, faces = vertices.cuda(), faces.cuda()
    inputs_optimizer = MeshOptimizer(vertices.detach(), faces.detach(), ramp=5,
                                     edge_len_lims=(end_edge_len, start_edge_len),
                                     local_edgelen=False)
    vertices = inputs_optimizer.vertices

    optim_epoch = 100
    batch_size = 8
    pbar = tqdm(range(optim_epoch))

    # Main optimization loop
    for iteration in pbar:
        perm = torch.randperm(num).cuda()
        # mean_ncc_loss = 0
        for k in range(0, batch_size):
            ref_w2c = w2cs[perm[k:k + 1]]
            ref_proj = projs[perm[k:k + 1]]
            # ref_gray = grayimgs[perm[k:k + 1]]
            ref_normal = gt_normals[perm[k:k + 1]]
            ref_mask = masks[perm[k:k + 1]]
            src_w2c = w2cs[pairs[perm[k]]]
            # src_pose = poses[pairs[perm[k]]]
            src_proj = projs[pairs[perm[k]]]
            # src_gray = grayimgs[pairs[perm[k]]]
            src_normal = gt_normals[pairs[perm[k]]]
            src_mask = masks[pairs[perm[k]]]

            depth = torch.cat([depths[perm[k:k + 1]], depths[pairs[perm[k]]]])
            w2c = torch.cat([ref_w2c, src_w2c])
            proj = torch.cat([ref_proj, src_proj])
            gt_normal = torch.cat([ref_normal, src_normal])
            # mask = torch.cat([ref_mask, src_mask])
            n = w2c.shape[0]

            # SAP generation
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:, 0:1])], axis=1).unsqueeze(0).expand(n, -1, -1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
            normals = get_normals(vertsw[:, :, :3], faces.long())

            int32_faces = faces.to(torch.int32)
            rast_out, _ = dr.rasterize(glctx, proj_verts, int32_faces, resolution=resolution)

            # render depth
            feat = torch.cat([rot_verts[:, :, :3], torch.ones_like(vertsw[:, :, :1]), vertsw[:, :, :3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, int32_faces)
            rast_verts = feat[:, :, :, :3].contiguous()
            pred_mask = feat[:, :, :, 3:4].contiguous()
            # rast_points = feat[:, :, :, 4:7].contiguous()
            # pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, int32_faces).squeeze(-1)

            # render normal
            feat, _ = dr.interpolate(normals, rast_out, int32_faces)
            pred_normals = feat.contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, int32_faces)
            pred_normals = F.normalize(pred_normals, p=2, dim=3)

            # # Compute Mask Loss
            # mask_loss = mask_weight * F.mse_loss(pred_mask, mask)

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
                print('Warning: normal_loss is None')
                normal_loss = torch.tensor(0.0, device=pred_normals.device)

                # Compute Normal L1 loss
            normal_l1_error = torch.abs((pred_normals - gt_normal[..., :3]))[gt_normal_mask]
            normal_l1_loss = normal_l1_error.mean() * normal_l1_weight

            # Compute Depth L1 loss
            depth_mask = depth != 0
            valid_depth = depth[depth_mask]
            render_depth = rast_verts[:, :, :, 2]
            valid_render_depth = render_depth.unsqueeze(1)[depth_mask.squeeze(-1)]
            depth_loss = torch.abs((valid_depth - valid_render_depth)).mean() * depth_weight

            # Optimizer step
            total_loss = (depth_loss + normal_loss + normal_l1_loss) / batch_size
            total_loss.backward()

        inputs_optimizer.step()
        inputs_optimizer.zero_grad()

        # Update progress bar description
        update_pbar_description_scene(pbar, depth_loss, normal_loss, normal_l1_loss)
        vertices, faces = inputs_optimizer.remesh()

        if iteration % 10 == 0:
            # Save intermediate results
            with torch.no_grad():
                np_vertices, np_faces = vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
                save_mesh = trimesh.Trimesh(np_vertices, np_faces, process=False, maintain_order=True)
                # save_mesh = clean_mesh(save_mesh, thresh=0.01)
                save_mesh.export(output_path)


def update_pbar_description_object(pbar, ncc_loss, mask_loss, sparse_loss):
    des = f'ncc:{ncc_loss.item():.4f} m:{mask_loss.item():.4f} normal_grad:{sparse_loss.item():.4f}'
    # des = f'm:{mask_loss.item():.4f} normal:{sparse_loss.item():.4f}'
    pbar.set_description(des)


def update_pbar_description_scene(pbar, depth_loss, normal_loss, normal_l1_loss):
    des = f'depth:{depth_loss.item():.4f} normal:{normal_loss.item():.4f} normal L1:{normal_l1_loss.item():.4f}'
    pbar.set_description(des)


@click.command()
@click.option('--source_path', '-s', type=str, help='Path to dataset')
@click.option('--model_path', '-m', type=str, help='Path to model')
@click.option('--output_path', '-o', type=str, help='Path to output')
@click.option('--num_points', default=None, type=int, help='Number of points')
@click.option('--num_sample', default=None, type=int, help='Number of samples')
@click.option('--h_patch_size', default=None, type=int, help='Patch size')
@click.option('--ncc_thresh', default=None, type=float, help='NCC threshold')
@click.option('--lr', default=None, type=float, help='Learning rate')
@click.option('--ncc_weight', default=None, type=float, help='NCC weight')
@click.option('--normal_weight', default=None, type=float, help='Normal weight')
@click.option('--normal_grad_weight', default=None, type=float, help='Normal gradient weight (for object)')
@click.option('--mask_weight', default=None, type=float, help='Mask weight')
@click.option('--normal_l1_weight', default=None, type=float, help='Normal L1 weight (for scene)')
@click.option('--depth_weight', default=None, type=float, help='Depth weight (for scene)')
@click.option('--atol', default=None, type=float, help='Tolerance level for alignment')
@click.option('--resolution', '-r', default=None, type=int, help='Resolution')
@click.option('--save_mid', default=None, type=int, help='Save intermediate results (for object)')
@click.option('--start_edge_len', '-sel', default=None, type=float, help='Start edge length for MeshOptimizer')
@click.option('--end_edge_len', '-eel', default=None, type=float, help='End edge length for MeshOptimizer')
@click.option('--laplacian_weight', '-lw', default=None, type=float, help='Laplacian weight for smoothing')
@click.option('--data_type', '-t', required=True, type=click.Choice(['scene', 'object']),
              help='Type of data to process')
def main(source_path, model_path, output_path, num_points, num_sample, h_patch_size, ncc_thresh, lr,
         ncc_weight, normal_weight, normal_grad_weight, mask_weight, normal_l1_weight, depth_weight,
         atol, resolution, save_mid, start_edge_len, end_edge_len, laplacian_weight, data_type):
    params = merge_params(
        data_type,
        num_points=num_points,
        num_sample=num_sample,
        h_patch_size=h_patch_size,
        ncc_thresh=ncc_thresh,
        lr=lr,
        ncc_weight=ncc_weight,
        normal_weight=normal_weight,
        normal_grad_weight=normal_grad_weight,
        mask_weight=mask_weight,
        normal_l1_weight=normal_l1_weight,
        depth_weight=depth_weight,
        atol=atol,
        resolution=resolution,
        save_mid=save_mid,
        start_edge_len=start_edge_len,
        end_edge_len=end_edge_len,
        laplacian_weight=laplacian_weight
    )

    click.echo(f"Processing {data_type} data with parameters:")
    for key, value in params.items():
        click.echo(f"{key} = {value}")

    if data_type == 'scene':
        refine_scene(
            source_path,
            model_path,
            output_path,
            params['num_points'],
            params['num_sample'],
            params['h_patch_size'],
            params['ncc_thresh'],
            params['lr'],
            params['ncc_weight'],
            params['normal_weight'],
            params['mask_weight'],
            params['normal_l1_weight'],
            params['depth_weight'],
            params['atol'],
            params['resolution'],
            params['start_edge_len'],
            params['end_edge_len'],
            params['laplacian_weight']
        )
    elif data_type == 'object':
        refine_object(
            source_path,
            model_path,
            output_path,
            params['num_points'],
            params['num_sample'],
            params['h_patch_size'],
            params['ncc_thresh'],
            params['lr'],
            params['ncc_weight'],
            params['normal_weight'],
            params['normal_grad_weight'],
            params['mask_weight'],
            params['atol'],
            params['resolution'],
            params['save_mid'],
            params['start_edge_len'],
            params['end_edge_len']
        )


if __name__ == '__main__':
    main()