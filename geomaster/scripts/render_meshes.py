import os
import torch
import argparse
import multiprocessing as mp
import torch.multiprocessing as tmp
import queue
import time
import subprocess

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


def process_ply_file(ply_path, gpu_id, output_dir, log_file, scale, res, num_round_views, num_elevations, min_elev, max_elev, space, render_mesh_script, process_idx):
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        render_mesh_script = os.path.join(script_dir, render_mesh_script)
                
        # Clear CUDA cache before rendering
        torch.cuda.empty_cache()
        
        # Build the command
        cmd = [
            "python", render_mesh_script,
            "--model_path", ply_path,
            "--output_dir", save_path,
            "--scale", str(scale),
            "--res", str(res),
            "--num_round_views", str(num_round_views),
            "--num_elevations", str(num_elevations),
            "--min_elev", str(min_elev),
            "--max_elev", str(max_elev),
            "--space", space,
            "--gpu_id", str(gpu_id)
        ]

        subprocess.run(cmd, check=True)
        # Clear CUDA cache after processing
        torch.cuda.empty_cache()

        # Log completion
        log_processed_id(log_file, obj_id)
        print(f"Processed {obj_id} on GPU {gpu_id} (Process {process_idx})")
        
    except Exception as e:
        print(f"Error processing {ply_path} on GPU {gpu_id} (Process {process_idx}): {str(e)}")
        # Clear CUDA cache in case of error
        torch.cuda.empty_cache()

def worker(task_queue, gpu_id, args, process_idx):
    """Worker process function."""
    try:
        while True:
            try:
                ply_path = task_queue.get_nowait()
                if ply_path is None:
                    break
                process_ply_file(
                    ply_path=ply_path,
                    gpu_id=gpu_id,
                    output_dir=args.output_dir,
                    log_file=args.log_file,
                    scale=args.scale,
                    res=args.res,
                    num_round_views=args.num_round_views,
                    num_elevations=args.num_elevations,
                    min_elev=args.min_elev,
                    max_elev=args.max_elev,
                    space=args.space,
                    render_mesh_script=args.render_mesh_script,
                    process_idx=process_idx,
                )
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
        # print(files)
        for file in files:
            if file.endswith('.ply'):
                ply_path = os.path.join(root, file)
                ply_files.append(ply_path)
    
    # Put all tasks in the queue
    for ply_path in ply_files:
        task_queue.put(ply_path)
    
    # Add termination signals
    for _ in range(args.num_workers):
        task_queue.put(None)
    
    # Create and start worker processes
    processes = []
    gpu_id = 0  # Using single GPU (cuda:0)
    
    for process_idx in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(task_queue, gpu_id, args, process_idx)
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
    parser.add_argument('--scale', '-s', type=float, default=2.5, help="Scale for rendering.")
    parser.add_argument('--res', '-r', type=int, default=512, help="Resolution of the rendered image.")
    parser.add_argument('--num_round_views', '-nrv', type=int, default=16, help="Number of round views.")
    parser.add_argument('--num_elevations', '-ne', type=int, default=4, help="Number of elevation angles.")
    parser.add_argument('--min_elev', '-min', type=float, default=-20, help="Minimum elevation angle.")
    parser.add_argument('--max_elev', '-max', type=float, default=40, help="Maximum elevation angle.")
    parser.add_argument('--space', '-sp', type=str, choices=['camera', 'world'], default='camera',
                        help="Space for normal rendering (camera or world). Default is 'camera'.")
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help="Number of parallel workers.")
    parser.add_argument('--render_mesh_script', '-rm', type=str, default='render_mesh.py',
                        help="Path to the render_mesh.py script.")
    
    args = parser.parse_args()
    main(args)

