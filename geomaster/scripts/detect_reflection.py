import click
import torch
import glob
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def classify_reflection(score_image, object_mask=None):
    score_array = torch.tensor(np.asarray(score_image) / 255)
    
    if object_mask is not None:
        mask_array = torch.tensor(np.asarray(object_mask) / 255)
        masked_score = score_array[mask_array > 0]
        mean_score = masked_score.mean()
    else:
        mean_score = score_array.mean()
    
    return mean_score.item()

def process_delight(image_path, output_delight_dir, input_mask_dir, delight_predictor):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image = Image.open(image_path)
       
    output_delight_path = os.path.join(output_delight_dir, f"{image_name}.png")
    if not os.path.exists(output_delight_path):
        with torch.inference_mode():
            delight_image = delight_predictor(input_image) 
        delight_image.save(output_delight_path)
    else:
        delight_image = Image.open(output_delight_path)
    
    output_mask_path = os.path.join(output_delight_dir, f"{image_name}_mask.png")
    score_image = delight_predictor.generate_reflection_score(input_image, delight_image)
    score_image.save(output_mask_path)
    
    object_mask_path = os.path.join(input_mask_dir, f"{image_name}.png")
    if os.path.exists(object_mask_path):
        object_mask = Image.open(object_mask_path)
    else:
        object_mask = None
    
    # Calculate the mean reflection score for the image
    mean_score = classify_reflection(score_image, object_mask)
    return mean_score
        
@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images dir')
@click.option('--masks', '-m', default="mask", help='Path to the masks dir')
@click.option('--num_workers', '-w', default=1, help='Number of worker threads')
@click.option('--threshold', '-t', default=0.3, help='Reflection classification threshold')
def main(source_path: str, images: str, masks: str, num_workers: int, threshold: float) -> None:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    delight_predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", trust_repo=True)
    
    input_mask_dir = os.path.join(source_path, masks)
    output_delight_dir = os.path.join(source_path, "delight")
    os.makedirs(output_delight_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(source_path, images, "*.png")) + \
                  glob.glob(os.path.join(source_path, images, "*.jpg"))
    
    scores = []
    for image_path in tqdm(image_paths):
        score = process_delight(image_path, output_delight_dir, input_mask_dir, delight_predictor)
        scores.append(score)
    
    # Calculate the average reflection score for the entire dataset
    avg_score = sum(scores) / len(scores)
    
    # Classify the dataset as reflection or non-reflection based on the average score
    if avg_score > threshold:
        print(f"The dataset is classified as reflection. Average score: {avg_score:.4f}")
    else:
        print(f"The dataset is classified as non-reflection. Average score: {avg_score:.4f}")

if __name__ == "__main__":
    main()