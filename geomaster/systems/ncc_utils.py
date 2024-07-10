import numpy as np
import torch

def NCC(ref, src, ref_valid_mask, src_valid_mask):
    '''
    ref 1 npoints npixels
    src nviews npoints npixels
    ref_valid_mask 1 npoints npixels
    src_valid_mask nviews npoints npixels
    
    return ncc nviews npoints
    '''
    nviews = src.shape[0]

    src_valid_num = torch.sum(src_valid_mask, dim=2, keepdim=True)
    src_valid_num[src_valid_num==0] = 1

    ref_mean = torch.sum(ref.expand(nviews,-1,-1)*src_valid_mask, dim=2, keepdim=True) / src_valid_num
    ref_var = torch.sum(((ref.expand(nviews,-1,-1)-ref_mean)*src_valid_mask).square(), dim=2, keepdim=True) / src_valid_num
    tmp = torch.zeros_like(ref_var)
    tmp[ref_var==0] = 1
    ref_var = tmp + ref_var
    ref_std = torch.sqrt(ref_var)

    src_mean = torch.sum(src*src_valid_mask, dim=2, keepdim=True) / src_valid_num
    src_var = torch.sum(((src-src_mean)*src_valid_mask).square(), dim=2, keepdim=True) / src_valid_num
    tmp = torch.zeros_like(src_var)
    tmp[src_var==0] = 1
    src_var = tmp + src_var
    src_std = torch.sqrt(src_var)

    cov = torch.sum((ref.expand(nviews,-1,-1)-ref_mean)*(src-src_mean) * src_valid_mask, dim=2, keepdim=True) / src_valid_num
    ncc = cov / (ref_std*src_std)

    return ncc.squeeze()

def normalize(flow, h, w, clamp=None):
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device

    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res

def add_hom(pts):
    try:
        dev = pts.device
        ones = torch.ones(pts.shape[:-1], device=dev).unsqueeze(-1)
        return torch.cat((pts, ones), dim=-1)

    except AttributeError:
        ones = np.ones((pts.shape[0], 1))
        return np.concatenate((pts, ones), axis=1)

def build_patch_offset(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 

def patch_homography(H, uv):
    '''
    H nview npoints 3 3
    uv npoints npixels 2
    '''
    N, Npx = uv.shape[:2]
    Nsrc = H.shape[0]
    H = H.view(Nsrc, N, 3, 3)
    hom_uv = add_hom(uv)

    tmp = torch.einsum("vpik,pok->vpoi", H, hom_uv).reshape(Nsrc, -1, 3)

    grid = tmp[..., :2] / torch.clamp(tmp[..., 2:], 1e-8)
    mask = tmp[..., 2] > 0
    return grid, mask