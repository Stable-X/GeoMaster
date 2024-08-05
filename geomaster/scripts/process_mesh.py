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
from gaustudio.cameras.camera_paths import get_path_from_orbit
import torchvision
from torch.optim import Adam
import numpy as np

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the model')
@click.option('--output_path', '-o', help='Path to the output mesh')
@click.option('--sap_res', default=512, type=int, help='SAP resolution')
@click.option('--num_sample', default=50000, type=int, help='Number of samples')
@click.option('--sig', default=2, type=int, help='Sigma value')
@click.option('--refine', default=False, type=bool, help='Whether to refine')
@click.option('--occ', default=False, type=bool, help='Whether to generate occ')
def main(model_path: str, output_path: str, sap_res: int, num_sample: int, sig: int = 2, refine: bool =False, occ: bool = False) -> None:
    if output_path is None:
        output_path = model_path[:-4]+'.refined.ply'
    # Get original mesh
    gt_vertices, gt_faces = get_mesh(model_path)
    gt_vertices, gt_faces = gt_vertices.cuda(), gt_faces.cuda()
    gt_vertsw = torch.cat([gt_vertices, torch.ones_like(gt_vertices[:,0:1])], axis=1).unsqueeze(0)
    gt_vertices = gt_vertices.unsqueeze(0)
    gt_normals = get_normals(gt_vertices[:,:,:3], gt_faces.long())
    
    # Initialize SAP
    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((sap_res, sap_res, sap_res), sig).cuda()
    glctx = dr.RasterizeGLContext()

    # Generate input mesh
    inputs, center, scale = get_sap(model_path, num_sample)
    inputs, center, scale = inputs.cuda(), center.cuda(), scale.cuda()
    inputs.requires_grad_(True)
    inputs_optimizer = Adam([{'params': inputs, 'lr': 0.01}])
    
    # Generate camera path
    cameras = get_path_from_orbit(center.cpu().numpy(), scale.cpu().numpy() * 1.5, elevation=0)

    # Cache SAP output
    vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
    vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0)
    normals = get_normals(vertsw[:,:,:3], faces.long())
    mesh = Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), 
                   vertex_normals=normals.detach().cpu().numpy())
    mesh.export(output_path)
    
    # Refine Poisson Mesh
    if refine:
        # Generate gt normal map
        for _id, _camera in enumerate(cameras):
            _camera = _camera.to("cuda")
            _w2c = _camera.extrinsics.T
            _proj = _camera.projection_matrix
            _pose = _w2c.permute(1, 0).contiguous()
            resolution = _camera.image_width, _camera.image_height
            
            rot_verts = torch.einsum('ijk,ikl->ijl', gt_vertsw, _w2c.unsqueeze(0))
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, _proj.unsqueeze(0))
            
            int32_faces = gt_faces.to(torch.int32)
            rast_out, _ = dr.rasterize(glctx, proj_verts, int32_faces, resolution=resolution)
            
            # render depth
            feat = torch.cat([rot_verts[:,:,:3], torch.ones_like(gt_vertsw[:,:,:1]), gt_vertsw[:,:,:3]], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, int32_faces)
            rast_verts = feat[:,:,:,:3].contiguous()
            pred_mask = feat[:,:,:,3:4].contiguous()
            rast_points = feat[:,:,:,4:7].contiguous()
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, int32_faces).squeeze(-1)
            
            # render normal
            feat, _ = dr.interpolate(gt_normals, rast_out, int32_faces)
            pred_normals = feat.contiguous()
            pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, int32_faces)
            pred_normals = F.normalize(pred_normals,p=2,dim=3)

            # Convert normal to view space        
            _camera.normal = _camera.worldnormal2normal(pred_normals[0])
            # _camera.normal = pred_normals[0]
            torchvision.utils.save_image( (-1 * _camera.normal.permute(2, 0, 1) + 1) / 2, 
                                         f"out/normal_{_id}.png")
            _camera.mask = pred_mask.squeeze(1).squeeze(-1)
            
            
        optim_epoch = 10
        batch_size = 8
        pbar = tqdm(range(optim_epoch))
        num = len(cameras)
        for i in pbar:
            perm = torch.randperm(num).cuda()
            for k in range(0, batch_size):
                _camera = cameras[perm[k]].to("cuda")
                _w2c = _camera.extrinsics.T
                _proj = _camera.projection_matrix
                _pose = _w2c.permute(1, 0).contiguous()
                resolution = _camera.image_width, _camera.image_height
                
                vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
                vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0)
                normals = get_normals(vertsw[:,:,:3], faces.long())

                rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, _w2c.unsqueeze(0))
                proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, _proj.unsqueeze(0))
                
                int32_faces = faces.to(torch.int32)
                rast_out, _ = dr.rasterize(glctx, proj_verts, int32_faces, resolution=resolution)
                
                # render normal
                feat, _ = dr.interpolate(normals, rast_out, int32_faces)
                pred_normals = feat.contiguous()
                pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, int32_faces)
                pred_normals = F.normalize(pred_normals,p=2,dim=3)

                
                # Compute Normal Loss
                gt_normal = _camera.normal2worldnormal().detach()
                # gt_normal = _camera.normal
                gt_mask = _camera.mask.detach()
                gt_normal_mask = (gt_mask > 0) & (rast_out[0,:,:,3] > 0)
                normal_error = (1 - (pred_normals * gt_normal).sum(dim=3))     
                normal_loss =  100 * normal_error[gt_normal_mask].mean()
                inputs_optimizer.zero_grad()
                normal_loss.backward()
                inputs_optimizer.step()

        mesh = Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy(), 
                    vertex_normals=normals.detach().cpu().numpy())
        mesh.export(output_path)
    if occ:
        vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, 0, 0.5)
        mesh = Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces.detach().cpu().numpy()) # vertices in [-0.5, 0.5]
        output_normalize_path = model_path[:-4]+'.normalized.ply'
        mesh.export(output_normalize_path)
        
        sample_points = np.random.uniform(low=-0.55, high=0.55, size=(100000, 3))
        sample_occ = check_mesh_contains(mesh, sample_points)[0]
        sample_occ = np.packbits(sample_occ.astype(bool))
        np.savez(os.path.join(os.path.dirname(model_path), 'points.npz'), points=sample_points, occupancies=sample_occ)

        surface_points, _ = mesh.sample(100000, return_index=True)
        np.savez(os.path.join(os.path.dirname(model_path), 'pointcloud.npz'), points=surface_points)

if __name__ == "__main__":
    main()