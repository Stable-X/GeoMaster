import os
import torch
import argparse
import numpy as np
import trimesh
from utils.func import load_obj, make_sparse_camera, calc_vertex_normals
from utils.render import nvdiffRenderer
from scipy.spatial.transform import Rotation

# ---- Config ----
scale = 2
cam_path = './cameras'
res = 512

def prepare_renderer(device):
    """Initialize the renderer."""
    mv, proj = make_sparse_camera(cam_path, scale, device=device)
    renderer = nvdiffRenderer(mv, proj, [res, res], device=device)
    return renderer, mv

def load_processed_ids(log_file):
    """Load the list of already processed IDs from the log file."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)

def log_processed_id(log_file, obj_id):
    """Log a processed ID to the log file."""
    with open(log_file, 'a') as f:
        f.write(f"{obj_id}\n")

def save_image(image, save_path, index):
    """Save a single image with the specified index."""
    filename = f"{index:03d}.png"
    full_path = os.path.join(save_path, filename)
    image_np = image.cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range
    image_np[..., 0] *= -1
    image_np = (image_np + 1) / 2
    
    # Convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    import cv2
    cv2.imwrite(full_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

def save_camera_matrix(matrix, save_path, index):
    """Save a single camera matrix with the specified index."""
    filename = f"{index:03d}.npy"
    full_path = os.path.join(save_path, filename)
    np.save(full_path, matrix.cpu().numpy())

def process_ply_file(ply_path, renderer, mv, output_dir, device, log_file):
    """Load .ply, calculate normals, render, and save results."""
    obj_id = os.path.basename(os.path.dirname(ply_path))
    
    # Skip if the ID is already logged
    processed_ids = load_processed_ids(log_file)
    if obj_id in processed_ids:
        print(f"Skipping {obj_id}, already processed.")
        return

    save_path = os.path.join(output_dir, obj_id)
    os.makedirs(save_path, exist_ok=True)

    # Load the .ply file and calculate normals
    mesh = trimesh.load(ply_path)
    
    vertices = np.array(mesh.vertices, dtype=np.float32)
    rotation = Rotation.from_euler('xyz', (0, 180, 0), degrees=True)
    rotation_matrix = rotation.as_matrix()
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    
    target_vertices = torch.tensor(rotated_vertices, dtype=torch.float32, device=device)
    
    
    target_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    target_normals = calc_vertex_normals(target_vertices, target_faces)

    # Render the normals for all views
    rendered_normals = renderer.render(target_vertices, target_faces, normals=target_normals)

    print(f"Rendered normals shape: {rendered_normals.shape}")
    print(f"Model-view matrix shape: {mv.shape}")

    num_views = mv.shape[0]
    for i in range(num_views):
        # Get the normal map for this view
        view_normal = rendered_normals[i]
        
        print(f"View normal shape: {view_normal.shape}")
        
        # The renderer might output in [0, 1] range, if so, convert to [-1, 1]
        if view_normal.min() >= 0:
            view_normal = view_normal * 2 - 1
        
        # Extract rotation part of the model-view matrix
        rotation = mv[i, :3, :3].to(device)
        print(f"Rotation matrix shape: {rotation.shape}")
        
        # Transform to camera space (only the first 3 channels)
        camera_space_normal = torch.matmul(view_normal[..., :3].reshape(-1, 3), rotation.t()).reshape(view_normal.shape[:-1] + (3,))
        
        print(f"Camera space normal shape: {camera_space_normal.shape}")
        
        # Normalize the camera space normals
        normal_norm = torch.norm(camera_space_normal, dim=2, keepdim=True)
        background_mask = normal_norm > 1.1
        camera_space_normal = camera_space_normal / normal_norm
        camera_space_normal[background_mask.squeeze(-1)] = 0
        
        # Save the camera space normal image
        save_image(camera_space_normal, save_path, i+1)
        save_camera_matrix(mv[i], save_path, i+1)

    # Log the processed ID
    log_processed_id(log_file, obj_id)
    print(f"Processed {obj_id}.")

def main(args):
    """Main function to process all .ply files."""
    device = torch.device('cuda:0')
    renderer, mv = prepare_renderer(device)

    # Process all .ply files in the input directory
    for root, dirs, files in os.walk(args.input_dir):
        if 'poisson_mesh.ply' in files:
            ply_path = os.path.join(root, 'poisson_mesh.ply')
            print(ply_path)
            process_ply_file(ply_path, renderer, mv, args.output_dir, device, args.log_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch render .ply files with logging.")
    parser.add_argument('--input_dir', '-i', type=str, required=True, 
                        help="Path to the directory containing subdirectories with .ply files.")
    parser.add_argument('--output_dir', '-o', type=str, required=True, 
                        help="Path to the directory to save rendered results.")
    parser.add_argument('--log_file', '-l', type=str, required=True, 
                        help="Path to the log file to record processed IDs.")

    args = parser.parse_args()
    main(args)