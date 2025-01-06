import os
import torch
import argparse
import numpy as np
import trimesh
from geomaster.utils.remesh_utils import calc_vertex_normals
from geomaster.utils.camera_utils import make_round_views
import nvdiffrast.torch as dr
from torchvision import transforms

# load model
dinov2_model = torch.hub.load(os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main'), 'dinov2_vitl14_reg', source='local',pretrained=True)
dinov2_model.eval().cuda()
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
n_patch = 518 // 14
batch_size = 8

def save_image(image, save_path, index, mask):
    """Save a single image with the specified index."""
    filename = f"{index:03d}.png"
    full_path = os.path.join(save_path, filename)
    image_np = image.cpu().numpy()
    mask_np = mask.cpu().numpy().squeeze(-1)
    # Convert from [-1, 1] to [0, 1] range
    image_np[..., 0] *= -1
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


def process_ply_file(ply_path, output_dir, scale, res, num_round_views, num_elevations, min_elev, max_elev, gpu_id):
    """Process a single PLY file."""
    # Set device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load and process mesh
    mesh = trimesh.load(ply_path)
    # Convert scene to mesh if necessary
    if hasattr(mesh, 'geometry'):
        # If it's a scene, get the first mesh
        # Assuming the scene has at least one mesh
        mesh_name = list(mesh.geometry.keys())[0]
        mesh = mesh.geometry[mesh_name]

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    mesh.apply_transform(transformation_matrix)

    vertices = np.array(mesh.vertices, dtype=np.float32)
    target_vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    target_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    target_normals = calc_vertex_normals(target_vertices, target_faces)

    # Render normals
    additional_elevations = np.random.uniform(min_elev, max_elev, num_elevations)
    mv, proj, ele = make_round_views(num_round_views, additional_elevations, scale)
    glctx = dr.RasterizeCudaContext()
    mvp = proj @ mv  # C,4,4
    vertices = target_vertices
    faces = target_faces.type(torch.int32)
    normals = target_normals
    image_size = [res, res]
    V = vertices.shape[0]
    vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
    vertices_clip = vert_hom @ mvp.transpose(-2,-1) #C,V,4
    rast_out,_ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False) #C,H,W,4
    vert_nrm = (normals+1)/2 if normals is not None else colors
    nrm, _ = dr.interpolate(vert_nrm, rast_out, faces) #C,H,W,3
    alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
    nrm = torch.concat((nrm,alpha),dim=-1) #C,H,W,4
    rendered_normals = dr.antialias(nrm, rast_out, vertices_clip, faces) #C,H,W,4
    rendered_normals = rendered_normals * 2 -1
    rendered_normals = rendered_normals[..., :3] * rendered_normals[..., 3:]
    rendered_normals = rendered_normals.permute(0, 3, 1, 2)

    num_all_views = mv.shape[0]
    for i in range(0, num_all_views, batch_size):
        batch_normals = rendered_normals[i:i+batch_size]
        batch_normals = transform(batch_normals)
        batch_features = dinov2_model(batch_normals, is_training=True)
        batch_patchtokens = batch_features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(batch_size, 1024, n_patch, n_patch)

    print(f"Processed {ply_path} successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render a single .ply file.")
    parser.add_argument('--model_path', '-m', type=str, required=True, help="Path to the .ply file to process.")
    parser.add_argument('--output_dir', '-o', type=str, required=True, help="Path to the output directory.")
    parser.add_argument('--scale', '-s', type=float, default=1.5, help="Scale for rendering.")
    parser.add_argument('--res', '-r', type=int, default=518, help="Resolution of the rendered image.")
    parser.add_argument('--num_round_views', '-nrv', type=int, default=16, help="Number of round views.")
    parser.add_argument('--num_elevations', '-ne', type=int, default=4, help="Number of elevation angles.")
    parser.add_argument('--min_elev', '-min', type=float, default=-45, help="Minimum elevation angle.")
    parser.add_argument('--max_elev', '-max', type=float, default=45, help="Maximum elevation angle.")
    parser.add_argument('--gpu_id', '-g', type=int, default=0, 
                        help="GPU ID to use for processing. Default is 0.")
                        
    args = parser.parse_args()
    process_ply_file(args.model_path, args.output_dir, args.scale, args.res, args.num_round_views, 
                     args.num_elevations, args.min_elev, args.max_elev, args.gpu_id)
