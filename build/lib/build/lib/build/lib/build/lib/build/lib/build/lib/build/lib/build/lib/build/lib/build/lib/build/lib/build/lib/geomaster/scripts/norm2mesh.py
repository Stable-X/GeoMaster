import argparse
import click
import torch
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from geomaster.models.mesh import MeshOptimizer
from geomaster.utils.mesh_utils import get_normals
from geomaster.utils.mesh_utils import space_carving
from geomaster.utils.remesh_utils import calc_vertex_normals
from geomaster.utils.camera_utils import get_cameras_from_json
from geomaster.utils.camera_utils import build_volumes_projections
from geomaster.utils.camera_utils import camera_intrinsic_to_opengl_projection
from geomaster.utils.func import load_and_split_image
from trimesh import Trimesh
import nvdiffrast.torch as dr
import torch.nn.functional as F
from gaustudio.cameras.camera_paths import get_path_from_json
from gaustudio.models import ShapeAsPoints
import torchvision
from torch.optim import Adam
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss


def main(images_path: str, camera_path: str, output_path: str, init: int = 1, num_views: int = 4) -> None:
    glctx = dr.RasterizeGLContext()
    
#        cameras = get_path_from_json(camera_path)
#        image_size = cameras[0].image_width, cameras[0].image_height
#        intrinsics = [camera.intrinsics for camera in cameras]
#        extrinsics = [camera.extrinsics for camera in cameras]

    intrinsics, extrinsics, image_size = get_cameras_from_json(camera_path)
    K = intrinsics[0].reshape(3, 3).cpu().numpy(camera_path)
    proj = torch.from_numpy(camera_intrinsic_to_opengl_projection(K, w=image_size[0], h=image_size[1], n=0.01, f=5.)).to(pipe_device)
    mv = torch.cat([torch.from_numpy(extrinsics), torch.Tensor([[0, 0, 0, 1]]).repeat(num_views, 1).unsqueeze(1)], axis=1).reshape(num_views, 4, 4).to(pipe_device)
    mvp = proj @ mv
    us, vs, in_regions = build_volumes_projections(extrinsics, intrinsics)
    
    if init:
        
        vertices, faces = camera_utils.space_carving(
            alpha, us, vs, in_regions, erosion=0, dilation=0, img_size=image_size[0])
        
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        
        _ = mesh.export(os.path.join(output_path, f"space_carving.obj"))

        mesh = mesh.simplify_quadric_decimation(face_count=1500)
        vertices, faces = mesh.vertices, mesh.faces
        vertices = torch.from_numpy(vertices).to(pipe_device).float()
        faces = torch.from_numpy(faces.copy()).to(pipe_device).long()
    else:
        vertices, faces = make_sphere(level=2, radius=.5)
    
    
    # opt = MeshOptimizer(vertices.detach(), faces.detach(), laplacian_weight=0.2)
    # vertices = opt.vertices
    sap = ShapeAsPoints.from_mesh(mesh_path=mesh.export(), config=None)  # Initialize SAP from mesh
    opt = Adam([sap._xyz], lr=0.01)  # Optimize the SAP representation

    optim_epoch = 10
    batch_size = 4
    pbar = tqdm(range(optim_epoch))
    num = len(cameras)
    
    # load gt nomals
    gt_normals = load_and_split_image(images_path) #(N, H, W, 3)
    valid = (gt_normals > 0.05).sum(axis=-1) > 0
    alpha, invalid = valid, ~valid
    gt_normals = np.concatenate([gt_normals, alpha[..., None]], axis=-1)
    
    # Generate current SAP representation's mesh
    vertices, faces, _ = sap.generate_mesh()
    
    for i in pbar:
    
        # render normal
        opt.zero_grad()
        
        # Render current mesh to get normal images
        normals = calc_vertex_normals(vertices, faces)
        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ mvp.transpose(-2,-1) #C,V,4
        
        # convert world space to camera space
        mv_no_trans = mv[:, :3, :3]
        normals_view = normals @ mv_no_trans.transpose(-2, -1)  # V,3 -> C,V,3
        
        rast_out,_ = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=False) #C,H,W,4
        vert_col = (normals+1)/2 #V,3
        col,_ = dr.interpolate(vert_col, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        col = torch.concat((col,alpha),dim=-1) #C,H,W,4
        pred_normals = dr.antialias(col, rast_out, vertices_clip, faces) #C,H,W,4
        
        # Compute Normal Loss
        normal_loss = (pred_normals[..., :3] - gt_normals[..., :3]).abs().mean()
        alpha_loss = (pred_normals[..., 3:4] - gt_normals[..., 3:4]).abs().mean()
        _mesh = Meshes(verts=[vertices], faces=[faces])
        normal_consist_loss = mesh_normal_consistency(_mesh)
        loss = normal_loss * 1. + alpha_loss * 1. + normal_consist_loss * 1e-1
        loss.backward()
        opt.step()
        
        # Generate current SAP representation's mesh
        vertices, faces, _ = sap.generate_mesh()

    mesh = Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), 
                vertex_normals=normals.detach().cpu().numpy())
    mesh.export(os.path.join(output_path, f"model.obj"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str)
    parser.add_argument('--camera_path', type=str)
    parser.add_argument('--output_path', type=str, default='outputs')
    parser.add_argument('--init', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=4)

    args = parser.parse_args()
    
    main(args.images_path, args.camera_path, args.output_path, args.init, args.num_views)
