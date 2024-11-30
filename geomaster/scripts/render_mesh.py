import os
import torch
import argparse
import numpy as np
import trimesh
from geomaster.utils.remesh_utils import calc_vertex_normals
from geomaster.utils.camera_utils import make_round_views
from scipy.spatial.transform import Rotation
import multiprocessing as mp
from functools import partial
import torch.multiprocessing as tmp
import queue
import time
import nvdiffrast.torch as dr
from matplotlib import image


# ---- Config ----
scale = 2.5
res = 512
num_round_views = 16   # elevation = 0
num_elevations = 4
min_elev = -20
max_elev = 40
PROCESSES_PER_GPU = 1  # Number of processes per GPU

    
def load_processed_ids(log_file):
    """Load the list of already processed IDs from the log file."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)

def log_processed_id(log_file, obj_id):
    """Log a processed ID to the log file in a thread-safe manner."""
    with mp.Lock():
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

def process_ply_file(ply_path, gpu_id, output_dir, log_file, process_idx):
    """Process a single PLY file using the specified GPU."""
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        obj_id = os.path.basename(os.path.dirname(ply_path))
        # up_dir = os.path.basename(os.path.dirname(os.path.dirname(ply_path)))
        # Skip if already processed
        processed_ids = load_processed_ids(log_file)
        if obj_id in processed_ids:
            print(f"Skipping {obj_id}, already processed.")
            return

        save_path = os.path.join(output_dir, obj_id)
        os.makedirs(save_path, exist_ok=True)

        # Load and process mesh
        mesh = trimesh.load(ply_path)
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
#        rotation = Rotation.from_euler('xyz',(0,180,0),degrees=True)
#        rotation_matrix = rotation.as_matrix()
#        rotated_vertices = np.dot(vertices, rotation_matrix.T)
        # Use a smaller batch size or process in chunks if memory is an issue
        target_vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
        target_faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
        target_normals = calc_vertex_normals(target_vertices, target_faces)

        # Clear CUDA cache before rendering
        torch.cuda.empty_cache()
        
        # Render normals
        additional_elevations = np.random.uniform(min_elev, max_elev, num_elevations)
        mv, proj, ele = make_round_views(num_round_views, additional_elevations, scale)
        glctx = dr.RasterizeGLContext()
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

        num_all_views = mv.shape[0]
        for i in range(num_all_views):
            view_normal = rendered_normals[i]
            
            if view_normal.min() >= 0:
                view_normal = view_normal * 2 - 1
            
            rotation = mv[i, :3, :3].to(device)
            camera_space_normal = torch.matmul(view_normal[..., :3].reshape(-1, 3), rotation.t()).reshape(view_normal.shape[:-1] + (3,))
            
            normal_norm = torch.norm(camera_space_normal, dim=2, keepdim=True)
            background_mask = normal_norm > 1.1
            camera_space_normal = camera_space_normal / normal_norm
            camera_space_normal[background_mask.squeeze(-1)] = 0
            camera_space_normal = -1*camera_space_normal
            save_image(camera_space_normal, save_path, i+1)
            save_camera_matrix(mv[i], save_path, i+1)

        # Clear CUDA cache after processing
        torch.cuda.empty_cache()

        # Log completion
        log_processed_id(log_file, obj_id)
        print(f"Processed {obj_id} on GPU {gpu_id} (Process {process_idx})")
        
    except Exception as e:
        print(f"Error processing {ply_path} on GPU {gpu_id} (Process {process_idx}): {str(e)}")
        # Clear CUDA cache in case of error
        torch.cuda.empty_cache()

def worker(task_queue, gpu_id, output_dir, log_file, process_idx):
    """Worker process function."""
    try:
        while True:
            try:
                ply_path = task_queue.get_nowait()
                if ply_path is None:
                    break
                process_ply_file(ply_path, gpu_id, output_dir, log_file, process_idx)
            except queue.Empty:
                break
    except Exception as e:
        print(f"Worker error on GPU {gpu_id} (Process {process_idx}): {str(e)}")
    finally:
        # Ensure CUDA cache is cleared when worker exits
        torch.cuda.empty_cache()

def main(args):
    """Main function with multi-process support on single GPU."""
    # Verify CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available")
    
    # Initialize multiprocessing with spawn method for CUDA compatibility
    tmp.set_start_method('spawn', force=True)
    
    # Create task queue
    task_queue = mp.Queue()
    
    # Collect all PLY files
    ply_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_path = os.path.join(root, file)
                ply_files.append(ply_path)
    
    # Put all tasks in the queue
    for ply_path in ply_files:
        task_queue.put(ply_path)
    
    # Add termination signals
    for _ in range(PROCESSES_PER_GPU):
        task_queue.put(None)
    
    # Create and start worker processes
    processes = []
    gpu_id = 0  # Using single GPU (cuda:0)
    
    for process_idx in range(PROCESSES_PER_GPU):
        p = mp.Process(
            target=worker,
            args=(task_queue, gpu_id, args.output_dir, args.log_file, process_idx)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch render .ply files with multi-processing on single GPU.")
    parser.add_argument('--input_dir', '-i', type=str, required=True, 
                        help="Path to the directory containing subdirectories with .ply files.")
    parser.add_argument('--output_dir', '-o', type=str, required=True, 
                        help="Path to the directory to save rendered results.")
    parser.add_argument('--log_file', '-l', type=str, required=True, 
                        help="Path to the log file to record processed IDs.")

    args = parser.parse_args()
    main(args)
