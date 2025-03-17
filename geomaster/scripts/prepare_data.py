import click
import torch
import glob
from PIL import Image
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from gaustudio import datasets, models
from gaustudio.pipelines import initializers
from geomaster.systems.tsdf_fusion_pipeline import tsdf_fusion
from geomaster.utils.depth_utils import read_depth_meter, rotate_mapping, reverse_rotate_mapping, convert_image, \
    convert_depth, load_depth_with_conf, compute_scale_and_shift, grad_descent
from promptda.promptda import PromptDA


def process_image(image_path, lidar_depth_path, output_normal_dir, output_mask_dir, output_depth_dir,
                  normal_predictor, mask_predictor, depth_predictor, rotate=None):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image = Image.open(image_path)
    input_depth = read_depth_meter(lidar_depth_path)
    if rotate is not None:
        input_image = Image.fromarray(cv2.rotate(np.array(input_image), rotate))
        input_depth = cv2.rotate(input_depth, rotate)

    output_normal_path = os.path.join(output_normal_dir, f"{image_name}.png")
    if not os.path.exists(output_normal_path) and normal_predictor is not None:
        normal_image = normal_predictor(input_image)
        if rotate is not None:
            reverse_rotate = reverse_rotate_mapping[rotate]
            normal_image = Image.fromarray(cv2.rotate(np.array(normal_image), reverse_rotate))
        normal_image.save(output_normal_path)

    output_mask_path = os.path.join(output_mask_dir, f"{image_name}.png")
    if not os.path.exists(output_mask_path) and mask_predictor is not None:
        mask = mask_predictor.infer_pil(input_image)
        if rotate is not None:
            reverse_rotate = reverse_rotate_mapping[rotate]
            mask = cv2.rotate(mask, reverse_rotate)
        mask = Image.fromarray(mask)
        mask.save(output_mask_path)

    output_depth_path = os.path.join(output_depth_dir, f"{image_name}.png")
    if not os.path.exists(output_depth_path) and depth_predictor is not None:
        color_image = convert_image(input_image)
        prompt_depth = convert_depth(input_depth)

        pred = depth_predictor.predict(color_image, prompt_depth).squeeze().cpu().numpy()

        H, W, _ = np.array(input_image).shape
        depth_array = cv2.resize(pred, (W, H), interpolation=cv2.INTER_AREA)
        if rotate is not None:
            depth_array = cv2.rotate(depth_array, reverse_rotate_mapping[rotate])
        depth_array = np.uint16(np.clip(depth_array * 1000, 0, 65535))
        cv2.imwrite(output_depth_path, depth_array)


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
            mask = lidar_depths != 0
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


@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images dir')
@click.option('--depths', '-d', default="depths", help='Path to the depths dir')
@click.option('--confidences', '-c', default="confidence", help='Path to the confidence dir')
@click.option('--normals', '-n', default="normals", help='Path to the normals dir')
@click.option('--masks', '-m', default="mask", help='Path to the masks dir')
@click.option('--mono_depths', '-md', default="mono_depths", help='Path to the mono depths dir')
@click.option('--num_workers', '-w', default=4, help='Number of worker threads')
@click.option('--vox_size', '-v', default=0.02, type=float, help='Voxel size for TSDF fusion')
@click.option('--align', '-a', type=click.Choice(['closed_form', 'grad_descent']),
              default='closed_form', help='Alignment method')
@click.option('--rotation', '-r', type=click.Choice(['None', '90', '180', '90_INV']),
              default='None', help='Rotation option')
def main(source_path: str, images: str, depths: str, confidences: str, normals: str, masks: str, mono_depths: str,
         num_workers: int, vox_size: float, align: str, rotation: str) -> None:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    # mask_predictor = torch.hub.load("aim-uofa/GenPercept", "GenPercept_Segmentation", trust_repo=True)
    mask_predictor = None
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
    # normal_predictor = None
    depth_predictor = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to("cuda").eval()

    output_normal_dir = os.path.join(source_path, normals)
    output_mask_dir = os.path.join(source_path, masks)
    output_depth_dir = os.path.join(source_path, mono_depths)
    output_depth_aligned_dir = os.path.join(source_path, mono_depths + "_aligned")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    os.makedirs(output_depth_aligned_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(source_path, images, "*.png")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpg")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpeg"))
    lidar_depth_paths = sorted(
        glob.glob(os.path.join(source_path, depths, "*.png"))
    )
    for idx in tqdm(range(len(image_paths)), desc="Processing images"):
        image_path = image_paths[idx]
        lidar_depth_path = lidar_depth_paths[idx]
        process_image(image_path, lidar_depth_path, output_normal_dir, output_mask_dir, output_depth_dir,
                      normal_predictor, mask_predictor, depth_predictor, rotate=rotate_mapping[rotation])

    confidence_paths = sorted(
        glob.glob(os.path.join(source_path, confidences, "*.png"))
    )
    mono_depth_paths = sorted(
        glob.glob(os.path.join(source_path, mono_depths, "*.png"))
    )
    process_depth_align(lidar_depth_paths, confidence_paths, mono_depth_paths, output_depth_aligned_dir,
                        align_method=align)

    dataset = datasets.make({
        "name": "colmap",
        "source_path": source_path,
        "masks": masks,
        "data_device": "cuda",
        "w_mask": True
    })

    tsdf_fusion(dataset, source_path, depth_dir=mono_depths + "_aligned", vox_size=vox_size)

    # initializer = initializers.make({"name":"VisualHull",
    #                                  "radius_scale": 2.5,
    #                                  "resolution": 256})
    # pcd = models.make("general_pcd")
    # initializer(pcd, dataset)
    #
    # visual_hull_path = os.path.join(source_path, 'visual_hull.ply')
    # shutil.copy(os.path.join(initializer.ws_dir, 'visual_hull.ply'), visual_hull_path)
    #
    # highlight_start = "\033[1;32m"
    # highlight_end = "\033[0m"
    # highlighted_command = f"{highlight_start}gm-recon -s {source_path} -m {visual_hull_path}{highlight_end}"
    # print(f"Done. Run {highlighted_command} to get the final result.")


if __name__ == "__main__":
    main()
