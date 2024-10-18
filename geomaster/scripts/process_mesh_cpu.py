import click
import os
import numpy as np
import trimesh
from pysdf import SDF
import logging

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the input model')
@click.option('--output_dir', '-o', required=True, help='Path to the output directory')
@click.option('--num_sample', default=100000, type=int, help='Number of samples for point cloud and SDF')
def main(model_path: str, output_dir: str, num_sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original mesh
    mesh = trimesh.load(model_path)
    
    # Ensure the mesh is watertight by keeping only the largest component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        logging.info(f"Original mesh had {len(components)} components. Keeping only the largest.")
        areas = np.array([c.area for c in components])
        mesh = components[areas.argmax()]
    
    # Center and scale the mesh
    mesh.vertices -= mesh.centroid
    scale = 1 / max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices *= scale
    
    # Invert the mesh if needed (assuming the original was oriented correctly)
    if mesh.volume < 0:
        mesh.invert()
    
    # Apply the transformation to align with desired orientation
    transform = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(transform)
    
    # Generate SDF samples
    sdf = SDF(mesh.vertices, mesh.faces)
    sample_points = np.random.uniform(low=-0.55, high=0.55, size=(num_sample, 3))
    sample_occ = sdf.contains(sample_points)
    sample_occ = np.packbits(sample_occ.astype(bool))
    np.savez(os.path.join(output_dir, 'points.npz'), points=sample_points, occupancies=sample_occ)
    
    # Generate surface point cloud
    surface_points, surface_indices = mesh.sample(num_sample, return_index=True)
    surface_normals = mesh.face_normals[surface_indices]
    np.savez(os.path.join(output_dir, 'pointcloud.npz'), points=surface_points, normals=surface_normals)
    
    logging.info(f"Processing complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()