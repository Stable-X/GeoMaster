import click
import torch
import numpy as np
import open3d as o3d
import nvdiffrast.torch as dr
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from PIL import Image
import torchvision
from geomaster.models.sap import PSR2Mesh, DPSR, sap_generate
from geomaster.models.sap import gen_inputs as get_sap
from geomaster.utils.mesh_utils import get_normals
from gaustudio.cameras.camera_paths import get_path_from_orbit

def load_textured_mesh_open3d(model_path, texture_path=None):
    mesh = o3d.io.read_triangle_mesh(model_path, True)
    
    vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).cuda()
    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32).cuda()
    
    if mesh.has_triangle_uvs():
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        vertex_uvs = triangle_uvs_to_vertex_uvs(triangle_uvs, faces.cpu().numpy())
        uvs = torch.tensor(vertex_uvs, dtype=torch.float32).cuda()
    else:
        print("Warning: Mesh does not have UV coordinates. Using random UVs.")
        uvs = torch.rand((vertices.shape[0], 2), dtype=torch.float32).cuda()
    
    if texture_path:
        texture = Image.open(texture_path)
        texture_tensor = torch.tensor(np.array(texture), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    elif mesh.has_textures():
        texture = mesh.textures[0]
        texture_tensor = torch.tensor(np.asarray(texture), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    else:
        print("Warning: No texture found. Using a default colored texture.")
        texture_tensor = torch.ones((1, 3, 256, 256), dtype=torch.float32).cuda()

    uvs = torch.cat((uvs, torch.zeros_like(uvs[..., :1]), torch.ones_like(uvs[..., :1])), dim=-1)
    return vertices, faces, uvs, texture_tensor.permute(0, 2, 3, 1).contiguous()

def triangle_uvs_to_vertex_uvs(triangle_uvs, faces):
    num_vertices = np.max(faces) + 1
    vertex_uvs = np.zeros((num_vertices, 2))
    vertex_uv_counts = np.zeros(num_vertices)

    for i in range(faces.shape[0]):
        for j in range(3):
            vertex_idx = faces[i, j]
            uv = triangle_uvs[i * 3 + j]
            vertex_uvs[vertex_idx] += uv
            vertex_uv_counts[vertex_idx] += 1

    # Average the UVs for vertices that appear in multiple triangles
    vertex_uvs /= np.maximum(vertex_uv_counts[:, np.newaxis], 1)

    return vertex_uvs

predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True,
                           local_cache_dir='/workspace/code/InverseRendering/StableNormal/weights')
def predict_normal_from_color(color):
    normal_image = predictor(color, mode='stable')
    normal = -1 * ((np.asarray(normal_image) / 255 * 2) - 1)
    normal = torch.tensor(normal, dtype=torch.float32).cuda()
    return F.normalize(normal, p=2, dim=2)

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the model')
@click.option('--texture_path', '-t', help='Path to the texture (optional if embedded in model)')
@click.option('--output_path', '-o', help='Path to the output mesh')
@click.option('--sap_res', default=256, type=int, help='SAP resolution')
@click.option('--num_sample', default=50000, type=int, help='Number of samples')
@click.option('--sig', default=2, type=int, help='Sigma value')
def main(model_path: str, texture_path: str, output_path: str, sap_res: int, num_sample: int, sig: int = 2) -> None:
    if output_path is None:
        output_path = model_path[:-4]+'.refined.ply'
    
    # Load textured mesh
    gt_vertices, gt_faces, gt_uvs, texture = load_textured_mesh_open3d(model_path, texture_path)
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
    
    cameras = get_path_from_orbit(center.cpu().numpy(), scale.cpu().numpy() * 2.5, elevation=-15)
    for _id, _camera in enumerate(tqdm(cameras)):
        _camera = _camera.to("cuda")
        _w2c = _camera.extrinsics.T
        _proj = _camera.projection_matrix
        resolution = (_camera.image_width, _camera.image_height)
        
        rot_verts = torch.einsum('ijk,ikl->ijl', gt_vertsw, _w2c.unsqueeze(0))
        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, _proj.unsqueeze(0))
        rast_out, _ = dr.rasterize(glctx, proj_verts, gt_faces, resolution=resolution)
        feat = torch.cat([rot_verts[:,:,:3], torch.ones_like(gt_vertsw[:,:,:1])], dim=2)
        feat, _ = dr.interpolate(feat, rast_out, gt_faces)
        pred_mask = feat[:,:,:,3:4].contiguous()
        pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, gt_faces)

        texc, _ = dr.interpolate(gt_uvs.unsqueeze(0), rast_out, gt_faces)
        color = dr.texture(texture.contiguous(), texc[..., :2].contiguous())
        color = color * pred_mask
        color = color + torch.ones_like(color) * (1 - pred_mask)
        torchvision.utils.save_image(color[0].permute(2, 0, 1), f"out/color_{_id}.png")
        _camera.normal = predict_normal_from_color(Image.open(f"out/color_{_id}.png"))
        torchvision.utils.save_image((-1 * _camera.normal.permute(2, 0, 1) + 1) / 2, f"out/normal_{_id}.png")
        _camera.mask = pred_mask.squeeze(1).squeeze(-1)
    
    psr2mesh = PSR2Mesh.apply
    dpsr = DPSR((sap_res, sap_res, sap_res), sig).cuda()
    vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
    vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0)
    normals = get_normals(vertsw[:,:,:3], faces.long())[0]

    initial_mesh = o3d.geometry.TriangleMesh()
    initial_mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
    initial_mesh.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy())
    initial_mesh.vertex_normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
    o3d.io.write_triangle_mesh(output_path, initial_mesh)
    
    optim_epoch = 10
    batch_size = 8
    pbar = tqdm(range(optim_epoch))
    num = len(cameras)
    for i in pbar:
        perm = torch.randperm(num).cuda()
        for k in range(0, num, batch_size):
            batch_indices = perm[k:k+batch_size]
            
            vertices, faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:,0:1])], axis=1).unsqueeze(0)
            normals = get_normals(vertsw[:,:,:3], faces.long())
            normals = torch.cat([normals, torch.zeros_like(normals[:, :, :1])], dim=2)
            total_loss = 0
            for idx in batch_indices:
                _camera = cameras[idx]
                _w2c = _camera.extrinsics.T
                _proj = _camera.projection_matrix
                resolution = (_camera.image_width, _camera.image_height)
                
                rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, _w2c.unsqueeze(0))
                proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, _proj.unsqueeze(0))
                
                rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=resolution)
                
                feat, _ = dr.interpolate(normals, rast_out, faces)
                pred_normals = feat[..., :3].contiguous()
                pred_mask = feat[..., 3:4].contiguous().squeeze(1).squeeze(-1)
                pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, faces)
                pred_normals = F.normalize(pred_normals, p=2, dim=3)

                pred_normals_cam = _camera.worldnormal2normal(pred_normals[0])
                torchvision.utils.save_image((-1 * pred_normals_cam.permute(2, 0, 1) + 1) / 2, f"out/render_normal_{idx}.png")
                
                gt_mask = _camera.mask.detach()                
                gt_normal = _camera.normal2worldnormal().detach()
                gt_normal_mask = (gt_mask > 0) & (rast_out[0,:,:,3] > 0)
                normal_error = (1 - (pred_normals * gt_normal).sum(dim=3))     
                total_loss += 0.1 * normal_error[gt_normal_mask].mean() + \
                                10 * F.mse_loss(pred_mask, gt_mask)
            
            total_loss /= len(batch_indices)
            inputs_optimizer.zero_grad()
            total_loss.backward()
            inputs_optimizer.step()

    refined_vertices, refined_faces, _, _, _ = sap_generate(dpsr, psr2mesh, inputs, center, scale)
    refined_normals = get_normals(refined_vertices.unsqueeze(0), refined_faces.long())[0]

    refined_mesh = o3d.geometry.TriangleMesh()
    refined_mesh.vertices = o3d.utility.Vector3dVector(refined_vertices.detach().cpu().numpy())
    refined_mesh.triangles = o3d.utility.Vector3iVector(refined_faces.detach().cpu().numpy())
    refined_mesh.vertex_normals = o3d.utility.Vector3dVector(refined_normals.detach().cpu().numpy())
    o3d.io.write_triangle_mesh(output_path, refined_mesh)

if __name__ == "__main__":
    main()