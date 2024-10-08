import click
import torch
import glob
import os
from tqdm import tqdm
from geomaster.models.sap import PSR2Mesh, DPSR, sap_generate
from geomaster.models.sap import gen_inputs as get_sap
from geomaster.models.mesh import gen_inputs as get_mesh
from geomaster.utils.mesh_utils import get_normals
from geomaster.utils.occupancy_utils import check_mesh_contains
from trimesh import Trimesh
import nvdiffrast.torch as dr
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
import open3d as o3d

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the model')
@click.option('--output_dir', '-o', help='Path to the output mesh')
@click.option('--sap_res', default=384, type=int, help='SAP resolution')
@click.option('--num_sample', default=50000, type=int, help='Number of samples')
@click.option('--sig', default=2, type=int, help='Sigma value')
def main(model_path: str, output_dir: str, sap_res: int, num_sample: int, sig: int = 2) -> None:
    os.makedirs(output_dir, exist_ok=True)

    clean_path = model_path[:-4]+'.clean.ply'
    logging.info("Cleaning mesh.")
    ratio_threshold = 0.5
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    print("Analyzing connected components...")
    (triangle_clusters, 
    cluster_n_triangles,
    cluster_area) = mesh.cluster_connected_triangles()
    
    print("Finding largest component...") 
    triangle_clusters = np.array(triangle_clusters)  
    cluster_n_triangles = np.array(cluster_n_triangles)
    
    largest_cluster_idx = cluster_n_triangles.argmax()
    largest_cluster_n_triangles = cluster_n_triangles[largest_cluster_idx]

    print(f"Largest component has {largest_cluster_n_triangles} triangles")
    
    triangles_keep_mask = np.zeros_like(triangle_clusters, dtype=np.int32)
    saved_clusters = []
    
    print("Removing small components...")
    for i, n_tri in enumerate(cluster_n_triangles):
        if n_tri > ratio_threshold * largest_cluster_n_triangles:
            saved_clusters.append(i)
            triangles_keep_mask += triangle_clusters == i
            
    triangles_to_remove = triangles_keep_mask == 0
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    
    print(f"Removed {triangles_to_remove.sum()} triangles")
    o3d.io.write_triangle_mesh(clean_path, mesh)
    model_path = clean_path
    
    
    # Get original mesh
    gt_vertices, gt_faces = get_mesh(model_path)
    gt_vertices, gt_faces = gt_vertices.cuda(), gt_faces.cuda()
    gt_vertsw = torch.cat([gt_vertices, torch.ones_like(gt_vertices[:,0:1])], axis=1).unsqueeze(0)
    gt_vertices = gt_vertices.unsqueeze(0)
    gt_normals = get_normals(gt_vertices[:,:,:3], gt_faces.long())
    
    rotation_matrix = torch.tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]]).to(gt_vertices.device, gt_vertices.dtype)
    gt_vertices = torch.matmul(gt_vertices, rotation_matrix)
    gt_normals = torch.matmul(gt_normals, rotation_matrix)
    
    
    # Initialize SAP
    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((sap_res, sap_res, sap_res), sig).cuda()
    glctx = dr.RasterizeGLContext()

    # Generate input mesh
    inputs, center, scale = get_sap(model_path, num_sample)
    inputs, center, scale = inputs.cuda(), center.cuda(), scale.cuda()
    inputs.requires_grad_(True)
    inputs_optimizer = Adam([{'params': inputs, 'lr': 0.01}])
    
    # Cache SAP output
    vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, 0, 0.5)
    vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0)    
    normals = get_normals(vertsw[:,:,:3], faces.long())    
    mesh = Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), 
                   vertex_normals=-normals.detach().cpu().numpy()) # vertices in [-0.5, 0.5]
    mesh.export(os.path.join(output_dir, 'poisson_mesh.ply'))
    
    from pysdf import SDF
    f = SDF(vertices.detach().cpu().numpy(), faces.detach().cpu().numpy())
    sample_points = np.random.uniform(low=-0.55, high=0.55, size=(100000, 3))
    sample_occ = f.contains(sample_points)
    sample_occ = np.packbits(sample_occ.astype(bool))
    np.savez(os.path.join(output_dir, 'points.npz'), points=sample_points, occupancies=sample_occ)

    surface_points, surface_indices = mesh.sample(100000, return_index=True)
    surface_normals = mesh.face_normals[surface_indices]
    np.savez(os.path.join(output_dir, 'pointcloud.npz'), points=surface_points, normals=surface_normals)
if __name__ == "__main__":
    main()