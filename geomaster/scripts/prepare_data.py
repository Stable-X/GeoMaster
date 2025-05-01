import click
import torch
import glob
from PIL import Image
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
from gaustudio import datasets, models
from gaustudio.pipelines import initializers
from RealESRGAN import RealESRGAN
from geomaster.systems.tsdf_fusion_pipeline import tsdf_fusion
from geomaster.utils.depth_utils import read_depth_meter, rotate_mapping, reverse_rotate_mapping, convert_image, \
    convert_depth, process_depth_align
from promptda.promptda import PromptDA


def process_image_object(image_path, output_normal_dir, output_mask_dir, output_edge_dir, normal_predictor, mask_predictor, superresolution):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image = Image.open(image_path)

    output_normal_path = os.path.join(output_normal_dir, f"{image_name}.png")
    if not os.path.exists(output_normal_path):

        if superresolution:
            sr_model = RealESRGAN("cuda:0", scale=4)
            sr_model.load_weights('weights/RealESRGAN_x4.pth', download=True)
            sr_input_image = sr_model.predict(np.array(input_image).astype(np.uint8))
        else:
            sr_input_image = input_image
        normal_image = normal_predictor(sr_input_image, data_type="object")
        # Resize normal_image to a quarter of its original size
        normal_image = normal_image.resize((normal_image.width // 4, normal_image.height // 4))
        normal_image.save(output_normal_path)

    output_mask_path = os.path.join(output_mask_dir, f"{image_name}.png")
    if not os.path.exists(output_mask_path) and mask_predictor is not None:
        mask = mask_predictor.infer_pil(input_image)
        mask = Image.fromarray(mask)
        mask.save(output_mask_path)

    output_edge_path = os.path.join(output_edge_dir, f"{image_name}.png")
    if not os.path.exists(output_edge_path):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(gray_image, 100, 200)
        cv2.imwrite(output_edge_path, edges)


def process_image_scene(image_path, lidar_depth_path, output_normal_dir, output_mask_dir, output_depth_dir,
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
@click.option('--align', '-a', type=click.Choice(['closed_form', 'grad_descent']), default='closed_form', help='Alignment method')
@click.option('--rotation', '-r', type=click.Choice(['None', '90', '180', '90_INV']), default='None', help='Rotation option')
@click.option('--superresolution', '-sr', default=1, help='Selector for whether to perform super-resolution processing')
@click.option('--data_type', '-t', required=True, type=click.Choice(['scene', 'object']), help='Type of data to process: scene or object')
def main(source_path: str, images: str, depths: str, confidences: str, normals: str, masks: str, mono_depths: str,
         num_workers: int, vox_size: float, align: str, rotation: str, superresolution: int, data_type: str) -> None:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    if data_type == 'scene':
        print("Processing scene data...")
        # mask_predictor = torch.hub.load("aim-uofa/GenPercept", "GenPercept_Segmentation", trust_repo=True)
        mask_predictor = None
        normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
        depth_predictor = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to("cuda").eval()

        output_depth_dir = os.path.join(source_path, mono_depths)
        output_depth_aligned_dir = os.path.join(source_path, mono_depths + "_aligned")
        os.makedirs(output_depth_dir, exist_ok=True)
        os.makedirs(output_depth_aligned_dir, exist_ok=True)

    elif data_type == 'object':
        print("Processing object data...")
        mask_predictor = torch.hub.load("aim-uofa/GenPercept", "GenPercept_Segmentation", trust_repo=True)
        normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')

        output_edge_dir = os.path.join(source_path, "edge")
        os.makedirs(output_edge_dir, exist_ok=True)

    output_normal_dir = os.path.join(source_path, normals)
    output_mask_dir = os.path.join(source_path, masks)
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(source_path, images, "*.png")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpg")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpeg")))

    if data_type == 'scene':
        lidar_depth_paths = sorted(glob.glob(os.path.join(source_path, depths, "*.png")))
        for idx in tqdm(range(len(image_paths)), desc="Processing images"):
            image_path = image_paths[idx]
            lidar_depth_path = lidar_depth_paths[idx]
            process_image_scene(image_path, lidar_depth_path, output_normal_dir, output_mask_dir, output_depth_dir,
                          normal_predictor, mask_predictor, depth_predictor, rotate=rotate_mapping[rotation])

        confidence_paths = sorted(glob.glob(os.path.join(source_path, confidences, "*.png")))
        mono_depth_paths = sorted(glob.glob(os.path.join(source_path, mono_depths, "*.png")))
        process_depth_align(lidar_depth_paths, confidence_paths, mono_depth_paths, output_depth_aligned_dir, align_method=align)
    elif data_type == 'object':
        for image_path in tqdm(image_paths, desc="Processing images"):
            process_image_object(image_path, output_normal_dir, output_mask_dir, output_edge_dir, normal_predictor, mask_predictor, superresolution)

    dataset = datasets.make({
        "name": "colmap",
        "source_path": source_path,
        "masks": masks,
        "data_device": "cuda",
        "w_mask": True
    })

    if data_type == 'scene':
        tsdf_fusion(dataset, source_path, depth_dir=mono_depths + "_aligned", vox_size=vox_size)
    elif data_type == 'object':
        initializer = initializers.make({"name":"VisualHull",
                                         "radius_scale": 2.5,
                                         "resolution": 256})
        pcd = models.make("general_pcd")
        initializer(pcd, dataset)

        visual_hull_path = os.path.join(source_path, 'visual_hull.ply')
        shutil.copy(os.path.join(initializer.ws_dir, 'visual_hull.ply'), visual_hull_path)

        highlight_start = "\033[1;32m"
        highlight_end = "\033[0m"
        highlighted_command = f"{highlight_start}gm-refine -s {source_path} -m {visual_hull_path}{highlight_end}"
        print(f"Done. Run {highlighted_command} to get the final result.")

if __name__ == "__main__":
    main()