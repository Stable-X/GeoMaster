import click
import torch
import glob
from PIL import Image
import os
from tqdm import tqdm
from rembg import remove

@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images')
def main(source_path: str, images: str) -> None:
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
    output_normal_dir = os.path.join(source_path, "normals")
    output_mask_dir = os.path.join(source_path, "masks")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_pattern = f"{source_path}/{images}/*.jpg"
    for image_path in tqdm(glob.glob(image_pattern, recursive=True)):
        image_name = os.path.basename(image_path.split("/")[-1]).split(".")[0]
        
        input_image = Image.open(image_path)
        normal_image = normal_predictor(input_image)
        normal_image.save(os.path.join(output_normal_dir, image_name+'.png'))
        
        output_mask_path = os.path.join(output_mask_dir, image_name+'.png')
        if not os.path.exists(output_mask_path):
            mask = remove(input_image, only_mask=True)
            mask.save(output_mask_path)


if __name__ == "__main__":
    main()