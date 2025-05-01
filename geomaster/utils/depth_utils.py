import cv2
import torch
import os
import numpy as np
from torch import Tensor
from PIL import Image

from promptda.utils.io_wrapper import to_tensor_func, ensure_multiple_of

rotate_mapping = {
    "None": None,
    "90": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "90_INV": cv2.ROTATE_90_COUNTERCLOCKWISE
}

reverse_rotate_mapping = {
    cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,
    cv2.ROTATE_180: cv2.ROTATE_180,
    cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE
}


def read_depth_meter(depth_path, scale=0.001):
    # Read depth_max from image
    depth_im = Image.open(depth_path)
    png_info = depth_im.info
    depth_im = np.asarray(depth_im).astype(np.float32)
    if "depth_max" in png_info:
        depth_max = float(png_info["depth_max"])
        depth_im = (depth_im / 65535.0) * depth_max
    else:
        depth_im = depth_im * scale

    return depth_im


def convert_image(image, to_tensor=True, max_size=1008, multiple_of=14):
    '''
    Load image from path and convert to tensor
    max_size // 14 = 0
    '''
    image = np.asarray(image).astype(np.float32)
    image = image / 255.

    max_size = max_size // multiple_of * multiple_of
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    tar_h = ensure_multiple_of(h * scale)
    tar_w = ensure_multiple_of(w * scale)
    image = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
    if to_tensor:
        return to_tensor_func(image).to("cuda")
    return image


def convert_depth(depth, to_tensor=True):
    '''
    Load depth from path and convert to tensor
    '''
    if to_tensor:
        return to_tensor_func(depth).to("cuda")
    return depth


def load_depth_with_conf(depth_path, conf_path=None, depth_size=(994, 738)):
    depth = read_depth_meter(depth_path)

    if conf_path is not None:
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        h, w = depth.shape[:2]
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_NEAREST)

        threshold_value = 200
        _, binary_conf = cv2.threshold(conf, threshold_value, 255, cv2.THRESH_BINARY)
        conf = binary_conf // 255

        depth = depth * conf

    depth = cv2.resize(depth, depth_size, interpolation=cv2.INTER_NEAREST)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    return depth


# copy from monosdf
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


# copy from dn-splatter
def grad_descent(
        mono_depth_tensors: torch.Tensor,
        sparse_depths: torch.Tensor,
        iterations: int = 1000,
        lr: float = 0.1,
        threshold: float = 0.0,
        device: str = 'cuda',
) -> tuple[Tensor, float]:
    """Align mono depth estimates with sparse depths.

    Args:
        mono_depth_tensors: mono depths
        sparse_depths: sparse sfm points
        iterations: number of gradient descent iterations
        lr: learning rate
        threshold: masking threshold of invalid depths. Default 0.
        device: tensor device

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    for idx in range(mono_depth_tensors.shape[0]):
        scale = torch.nn.Parameter(
            torch.tensor([1.0], device=device, dtype=torch.float)
        )
        shift = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float)
        )

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_depth = sparse_depths[idx].float().to(device)

        mask = sparse_depth > threshold
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)

        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked + shift, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()
        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth + shift)

    avg = sum(avg_err) / len(avg_err)
    return torch.stack(aligned_mono_depths, dim=0), avg


def process_depth_align(lidar_depth_paths, confidence_paths, mono_depth_paths, output_depth_aligned_dir,
                        align_method="closed_form", batch_size=64):
    num_frames = len(lidar_depth_paths)

    for batch_index in range(0, num_frames, batch_size):
        batch_lidar_depth = lidar_depth_paths[
                            batch_index: batch_index + batch_size
                            ]
        if len(confidence_paths) != 0:
            batch_conf = confidence_paths[
                         batch_index: batch_index + batch_size
                         ]
        batch_mono_depth = mono_depth_paths[
                           batch_index: batch_index + batch_size
                           ]

        with torch.no_grad():
            mono_depth_tensors = []
            lidar_depths = []

            for frame_index in range(len(batch_lidar_depth)):
                sfm_frame = batch_lidar_depth[frame_index]
                if len(confidence_paths) != 0:
                    conf_frame = batch_conf[frame_index]
                else:
                    conf_frame = None
                mono_frame = batch_mono_depth[frame_index]

                mono_depth = load_depth_with_conf(mono_frame)
                mono_depth_tensors.append(mono_depth)

                sfm_depth = load_depth_with_conf(sfm_frame, conf_frame,
                                                 depth_size=(mono_depth.shape[1], mono_depth.shape[0]))
                lidar_depths.append(sfm_depth)

            mono_depth_tensors = torch.stack(mono_depth_tensors, dim=0)
            lidar_depths = torch.stack(lidar_depths, dim=0)

        if align_method == "closed_form":
            mask = (lidar_depths != 0) & (lidar_depths < 3.0)
            scale, shift = compute_scale_and_shift(mono_depth_tensors, lidar_depths, mask=mask)
            scale = scale.unsqueeze(1).unsqueeze(2)
            shift = shift.unsqueeze(1).unsqueeze(2)
            depth_aligned = scale * mono_depth_tensors + shift

            mse_loss = torch.nn.MSELoss()
            avg = mse_loss(depth_aligned[mask], lidar_depths[mask])
        elif align_method == "grad_descent":
            depth_aligned, avg = grad_descent(
                mono_depth_tensors, lidar_depths
            )
        else:
            raise NotImplementedError
        print(
            f"Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg < 0.2 else 'bad'}"
        )

        # save depths
        for idx in range(depth_aligned.shape[0]):
            depth_aligned_numpy = depth_aligned[idx, ...].detach().cpu().numpy()
            depth_array = np.uint16(np.clip(depth_aligned_numpy * 1000, 0, 65535))

            file_name = os.path.basename(batch_mono_depth[idx])
            cv2.imwrite(os.path.join(output_depth_aligned_dir, file_name), depth_array)
