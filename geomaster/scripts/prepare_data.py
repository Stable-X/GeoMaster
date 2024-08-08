import click
import torch
import glob
from PIL import Image
import os
import shutil
from tqdm import tqdm
from rembg import remove
from gaustudio import datasets, models
from gaustudio.pipelines import initializers

@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images')
def main(source_path: str, images: str) -> None:
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
    output_normal_dir = os.path.join(source_path, "normals")
    output_mask_dir = os.path.join(source_path, "mask")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_pattern = f"{source_path}/{images}/*.png"
    for image_path in tqdm(glob.glob(image_pattern, recursive=True)):
        image_name = os.path.basename(image_path.split("/")[-1]).split(".")[0]
        input_image = Image.open(image_path)
        
        output_normal_path = os.path.join(output_normal_dir, image_name + '.png')
        if not os.path.exists(output_normal_path):
            normal_image = normal_predictor(input_image)
            normal_image.save(output_normal_path)
            
        output_mask_path = os.path.join(output_mask_dir, image_name + '.png')
        if not os.path.exists(output_mask_path):
            mask = remove(input_image, only_mask=True)
            mask.save(output_mask_path)

    dataset = datasets.make({"name": "colmap", "source_path": source_path, "masks": 'mask', 
                             "data_device": "cuda", "w_mask": True})
    initializer = initializers.make("VisualHull")
    pcd = models.make("general_pcd")
    initializer(pcd, dataset)
    shutil.copy(os.path.join(initializer.ws_dir, 'visual_hull.ply'), os.path.join(source_path, 'visual_hull.ply'))
    print(f"Done. Run gm-process -s {source_path} -m {source_path}/visual_hull.ply to get the final result.")
if __name__ == "__main__":
    main()