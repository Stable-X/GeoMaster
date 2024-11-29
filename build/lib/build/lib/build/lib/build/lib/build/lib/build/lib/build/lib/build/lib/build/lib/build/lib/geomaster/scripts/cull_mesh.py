import click
import torch
from geomaster.models.mesh import gen_inputs as get_mesh
from gaustudio import datasets
from trimesh import Trimesh
from tqdm import tqdm 

@click.command()
@click.option('--model_path', '-m', required=True, help='Path to the model')
@click.option('--source_path', '-s', required=True, help='Path to the source dir')
@click.option('--output_path', '-o', help='Path to the output mesh')
@click.option('--min_weight', '-w', type=int, default=5, help="Min visible weight")
@click.option('--use_mask', is_flag=True, help="Use mask to prune")
def main(model_path: str, source_path:str, output_path: str, min_weight: int, use_mask: bool) -> None:
    if output_path is None:
        output_path = model_path[:-4]+'.clean.ply'

    # Get original mesh
    gt_vertices, gt_faces = get_mesh(model_path)
    gt_vertices, gt_faces = gt_vertices.cuda(), gt_faces.cuda()
    gt_vertices_weights = torch.zeros((gt_vertices.shape[0]), dtype=int).cuda()
    
    # Load cameras
    cameras = datasets.make({"name":"colmap", "source_path": source_path, 
                             "masks": 'mask', "w_mask": use_mask}).all_cameras
    
    # Calculate weights
    for _camera in tqdm(cameras):
        _camera = _camera.to("cuda")
        if use_mask:
            _inView = _camera.insideView(gt_vertices, _camera.mask)
        else:
            _inView = _camera.insideView(gt_vertices)
        gt_vertices_weights += _inView.int()

    # Prune vertices
    visible_vertices_mask = gt_vertices_weights >= min_weight
    pruned_vertices = gt_vertices[visible_vertices_mask]
    
    # Update faces
    vertex_map = torch.cumsum(visible_vertices_mask.int(), dim=0) - 1
    valid_faces_mask = visible_vertices_mask[gt_faces[:, 0]] & \
                       visible_vertices_mask[gt_faces[:, 1]] & \
                       visible_vertices_mask[gt_faces[:, 2]]
    pruned_faces = vertex_map[gt_faces[valid_faces_mask]]
    
    
    mesh = Trimesh(vertices=pruned_vertices.cpu().numpy(), faces=pruned_faces.cpu().numpy())
    mesh.export(output_path)
    
    print(f"Pruned mesh saved to {output_path}")
    print(f"Original vertices: {gt_vertices.shape[0]}, Pruned vertices: {pruned_vertices.shape[0]}")
    print(f"Original faces: {gt_faces.shape[0]}, Pruned faces: {pruned_faces.shape[0]}")

if __name__ == "__main__":
    main()