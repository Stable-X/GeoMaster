import click
import torch
import os
import open3d as o3d
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

def mesh_sap(pcd):
    from gaustudio.models import ShapeAsPoints
    sap_pcd = ShapeAsPoints.from_o3d_pointcloud(pcd)
    mesh = sap_pcd.to_o3d_mesh()
    return mesh

def gen_inputs(npz_file, num_sample=100000):
    pointcloud = np.load(npz_file)
    surface = np.asarray(pointcloud['points'])
    normal = np.asarray(pointcloud['normals'])

    # Convert to torch tensors
    surface = torch.from_numpy(surface.astype(np.float32))
    normal = torch.from_numpy(normal.astype(np.float32))

    # Normalize surface points to [0, 1]
    surface = (surface + 1) / 2
    print(surface.max(), surface.min())
    
    # Sample points if necessary
    if surface.shape[0] > num_sample:
        idx = np.random.choice(surface.shape[0], num_sample, replace=False)
        surface = surface[idx]
        normal = normal[idx]

    # Reshape tensors
    surface = surface.unsqueeze(0)
    normal = normal.unsqueeze(0)

    # Apply inverse sigmoid to surface points
    surface = torch.log(surface / (1 - surface))

    # Combine surface and normal data
    inputs = torch.cat([surface, normal], axis=-1)

    return inputs

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the input point cloud file')
@click.option('--output_dir', '-o', default='.', help='Path to the output directory')
@click.option('--num_sample', default=100000, type=int, help='Number of samples')
def main(model_path: str, output_dir: str, num_sample: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate input point cloud
    inputs = gen_inputs(model_path, num_sample)

    # Convert inputs to numpy array and create Open3D point cloud
    points = inputs[0, :, :3].cpu().numpy()
    normals = inputs[0, :, 3:].cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Convert point cloud to mesh using SAP
    mesh = mesh_sap(pcd)

    # Save the mesh as poisson_mesh.ply
    output_path = os.path.join(output_dir, "poisson_mesh.ply")
    o3d.io.write_triangle_mesh(output_path, mesh)
    logging.info(f"Mesh saved to {output_path}")

if __name__ == "__main__":
    main()