import os
import torch
import argparse
import numpy as np
import trimesh
from geomaster.utils.remesh_utils import calc_vertex_normals
from geomaster.utils.camera_utils import make_round_views
import nvdiffrast.torch as dr
from torchvision import transforms
import open3d as o3d
import torch.nn.functional as F
import click


def save_image(image, save_path, index, mask):
    """Save a single image with the specified index."""
    filename = f"{index:03d}.png"
    full_path = os.path.join(save_path, filename)
    image_np = image.cpu().numpy()
    mask_np = mask.cpu().numpy().squeeze(-1)
    # Convert from [-1, 1] to [0, 1] range
    image_np = (image_np + 1) / 2
    image_np[mask_np < 0.5] = 0
    # Convert to uint8
    image_np = (image_np * 255).astype(np.uint8)

    import cv2
    cv2.imwrite(full_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))


def save_camera_matrix(matrix, save_path, index):
    """Save a single camera matrix with the specified index."""
    filename = f"{index:03d}.npy"
    full_path = os.path.join(save_path, filename)
    np.save(full_path, matrix.cpu().numpy())

def aggregate_voxel_features(patch_tokens, uv_coords, device='cuda'):
    """
    Aggregate features from patch tokens to voxels using UV coordinates
    
    Args:
        patch_tokens: Tensor of shape (num_views, feature_dim, patch_height, patch_width)
        uv_coords: Tensor of shape (num_views, num_voxels, 2) containing UV coordinates
        device: Device to process on
    
    Returns:
        voxel_features: numpy array of shape (num_voxels, feature_dim) containing averaged features
    """
    # Move inputs to device if needed
    patch_tokens = patch_tokens.to(device)
    uv_coords = uv_coords.to(device)
    uv_normalized = uv_coords
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_features = []
    
    for i in range(0, patch_tokens.shape[0], batch_size):
        batch_end = min(i + batch_size, patch_tokens.shape[0])
        
        # Get current batch
        batch_tokens = patch_tokens[i:batch_end]  # (B, C, H, W)
        batch_uv = uv_normalized[i:batch_end]     # (B, V, 2)
        
        # Sample features using grid_sample
        # Need to add an extra dimension for grid_sample
        sampled_features = F.grid_sample(
            batch_tokens,
            batch_uv.unsqueeze(1),  # Add height dimension
            mode='bilinear',
            align_corners=False
        )  # (B, C, 1, V)
        
        # Remove extra dimensions and rearrange
        sampled_features = sampled_features.squeeze(2)  # (B, C, V)
        sampled_features = sampled_features.permute(0, 2, 1)  # (B, V, C)
        
        all_features.append(sampled_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)  # (num_views, num_voxels, feature_dim)
    
    # Average across views
    voxel_features = torch.mean(all_features, dim=0).numpy()  # (num_voxels, feature_dim)
    
    # Convert to float16 for memory efficiency
    voxel_features = voxel_features.astype('float16')
    
    return voxel_features

@click.command()
@click.option('--model-path', '-m', required=True, type=str, help="Path to the.ply file to process.")
@click.option('--output-dir', '-o', required=True, type=str, help="Path to the output directory.")
@click.option('--scale', '-s', default=1.5, type=float, help="Scale for rendering.")
@click.option('--res', '-r', default=518, type=int, help="Resolution of the rendered image.")
@click.option('--num-round-views', '-nrv', default=16, type=int, help="Number of round views.")
@click.option('--num-elevations', '-ne', default=4, type=int, help="Number of elevation angles.")
@click.option('--min-elev', '-min', default=-45, type=float, help="Minimum elevation angle.")
@click.option('--max-elev', '-max', default=45, type=float, help="Maximum elevation angle.")
@click.option('--gpu-id', '-g', default=0, type=int, help="GPU ID to use for processing. Default is 0.")
@click.option('--voxel_resolution', '-v', default=64, type=int)
def main(model_path, output_dir, scale, res, num_round_views, num_elevations, min_elev, max_elev, gpu_id, voxel_resolution):
    # Set device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # load model
    dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), 'dinov2_vitl14_reg', source='local',pretrained=True)
    dinov2_model.eval().to(device)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14
    batch_size = 8
    torch.set_grad_enabled(False)

    # Load and process mesh
    mesh = trimesh.load(model_path)
    # Convert scene to mesh if necessary
    if hasattr(mesh, 'geometry'):
        mesh_name = list(mesh.geometry.keys())[0]
        mesh = mesh.geometry[mesh_name]

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    inverse_rotation_matrix = transformation_matrix.T

    # Voxelization
    vertices_np = np.array(mesh.vertices, dtype=np.float32).clip(-0.5, 0.495)

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh_o3d,
        voxel_size=1/voxel_resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5)
    )
    voxel_vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    print(voxel_vertices.max(), voxel_vertices.min())
    assert np.all(voxel_vertices >= 0) and np.all(voxel_vertices < voxel_resolution), "Some vertices are out of bounds"
    
    positions = (voxel_vertices + 0.5) / voxel_resolution - 0.5
    positions = torch.from_numpy(positions).float().to(device)
    indices = ((positions + 0.5) * voxel_resolution).long()

    # Render normals
    target_vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device)
    target_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    target_normals = calc_vertex_normals(target_vertices, target_faces)

    additional_elevations = np.random.uniform(min_elev, max_elev, num_elevations)
    mv, proj, ele = make_round_views(num_round_views, additional_elevations, scale, device=device)
    inverse_rotation_matrix_tensor = torch.tensor(inverse_rotation_matrix, dtype=torch.float32, device=mv.device)
    mv = mv @ inverse_rotation_matrix_tensor

    # Convert vertex colors to tensor
    glctx = dr.RasterizeCudaContext(device)
    mvp = proj @ mv  # C,4,4

    # Calculate UV coordinates using MVP matrix
    V = positions.shape[0]
    pos_hom = torch.cat((positions, torch.ones(V, 1, device=positions.device)), dim=-1)  # V,4
    all_uvs = []
    
    for i in range(0, mvp.shape[0], batch_size):
        batch_end = min(i + batch_size, mvp.shape[0])
        batch_mvp = mvp[i:batch_end]  # B,4,4
        # Project points to clip space
        clip_coords = pos_hom @ batch_mvp.transpose(-2, -1)  # B,V,4
        
        # Perspective divide
        ndc = clip_coords[..., :3] / clip_coords[..., 3:4]  # B,V,3
        
        # Convert to UV coordinates (only take x,y components)
        batch_uv = ndc[..., :2]  # B,V,2
        all_uvs.append(batch_uv)
    all_uvs = torch.cat(all_uvs, dim=0)  # C,V,2

    vertices = target_vertices
    faces = target_faces.type(torch.int32)
    normals = target_normals
    image_size = [res, res]
    vert_hom = torch.cat((vertices, torch.ones(vertices.shape[0], 1, device=vertices.device)), dim=-1)
    vertices_clip = vert_hom @ mvp.transpose(-2, -1)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False)
    vert_nrm = normals
    nrm, _ = dr.interpolate(vert_nrm, rast_out, faces)
    alpha = torch.clamp(rast_out[..., -1:], max=1)
    nrm = torch.concat((nrm, alpha), dim=-1)
    rendered_normals = dr.antialias(nrm, rast_out, vertices_clip, faces)
    rendered_normals = rendered_normals[..., :3] * rendered_normals[..., 3:]
    rendered_normals = rendered_normals.permute(0, 3, 1, 2)

    all_patchtokens = []
    for i in range(0, mvp.shape[0], batch_size):
        batch_end = min(i + batch_size, mvp.shape[0])
        batch_normals = (rendered_normals[i:batch_end] + 1)/2 # [0, 1]
        batch_normals = transform(batch_normals)
        batch_features = dinov2_model(batch_normals, is_training=True)
        batch_patchtokens = batch_features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(batch_end - i, 1024, n_patch, n_patch)
        all_patchtokens.append(batch_patchtokens)

    all_patchtokens = torch.cat(all_patchtokens, dim=0)
    voxel_features = aggregate_voxel_features(all_patchtokens,all_uvs)
    voxel_normals = aggregate_voxel_features(rendered_normals,all_uvs)

    os.makedirs(output_dir, exist_ok=True)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(positions.cpu().numpy())
    point_cloud.normals = o3d.utility.Vector3dVector(voxel_normals)
    o3d.io.write_point_cloud(os.path.join(output_dir, 'voxel_points.ply'), point_cloud)

    indices_np = indices.cpu().numpy()
    indices_np = np.concatenate([np.zeros((indices_np.shape[0], 1)), indices_np], axis=1).astype(np.int32)
    np.savez(os.path.join(output_dir, 'dino_features.npz'), feats=voxel_features, coords=indices_np)
