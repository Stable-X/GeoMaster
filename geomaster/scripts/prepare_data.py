import click
import torch
import glob
from PIL import Image
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from gaustudio import datasets, models
from gaustudio.pipelines import initializers
from RealESRGAN import RealESRGAN
import numpy as np
import cv2

            
def process_image(image_path, output_normal_dir, output_mask_dir, output_edge_dir, normal_predictor, mask_predictor, superresolution):
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
    if not os.path.exists(output_mask_path):
        mask = mask_predictor.infer_pil(input_image)
        mask = Image.fromarray(mask)
        mask.save(output_mask_path)
    
    output_edge_path = os.path.join(output_edge_dir, f"{image_name}.png")
    if not os.path.exists(output_edge_path):
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(gray_image, 100, 200)
        cv2.imwrite(output_edge_path, edges)

@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images dir')
@click.option('--masks', '-m', default="mask", help='Path to the masks dir')
@click.option('--num_workers', '-w', default=4, help='Number of worker threads')
@click.option('--superresolution', '-sr', default=1, help='Selector for whether to perform super-resolution processing')
def main(source_path: str, images: str, masks: str, num_workers: int, superresolution: int) -> None:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    #mask_predictor = torch.hub.load("aim-uofa/GenPercept", "GenPercept_Segmentation", trust_repo=True)
    import sys
    local_models_dir = "/home/jiahao/.cache/torch/hub/aim-uofa_GenPercept_main/GenPercept_v1/"
    sys.path.append(local_models_dir)
    from gp_hubconf import GenPercept_Segmentation
    mask_predictor = GenPercept_Segmentation()
    normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')
    
    output_normal_dir = os.path.join(source_path, "normals")
    output_mask_dir = os.path.join(source_path, masks)
    output_edge_dir = os.path.join(source_path, "edge")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_edge_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(source_path, images, "*.png")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpg")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpeg"))

    for image_path in tqdm(image_paths, desc="Processing images"):
        process_image(image_path, output_normal_dir, output_mask_dir, output_edge_dir, normal_predictor, mask_predictor, superresolution)

    dataset = datasets.make({
        "name": "colmap", 
        "source_path": source_path, 
        "masks": masks, 
        "data_device": "cuda", 
        "w_mask": True
    })
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