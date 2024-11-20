import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from skimage import measure
from scipy import ndimage


def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces] # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2, eps=1e-6)
    # verts_normals = mynormalize(verts_normals, p=2, dim=2, eps=1e-6)

    return verts_normals
    
    
def space_carving(alphas, us, vs, in_regions, resolution=64, erosion=0, dilation=0, img_size=256):
    num_cams = alphas.shape[0]
    valids = np.zeros([num_cams, resolution * resolution * resolution], np.float32)
    # valids = np.ones([num_cams, resolution * resolution * resolution], np.float32)
    for i in range(num_cams):
        alpha, u, v, in_region = alphas[i], us[i], vs[i], in_regions[i]
        values = alpha[img_size - 1 - v[in_region], u[in_region]]
        valids[i][~in_region] = 0
        valids[i][in_region] = values
    valids = valids.sum(axis=0) == num_cams
    valids = valids.reshape(resolution, resolution, resolution)
    # dilation for loose supervision
    if dilation > 0:
        valids = ndimage.binary_dilation(valids, iterations=dilation)
    if erosion > 0:
        valids = ndimage.binary_erosion(valids, iterations=erosion)

    verts, faces, normals, values = measure.marching_cubes(valids, 0.5, gradient_direction='ascent')
    verts = verts / (resolution - 1.) - 0.5
    
    return verts, faces