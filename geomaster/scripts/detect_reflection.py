import click
import torch
import glob
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

def generate_specular(rgb_image, diffuse_image, kernel_size=15, threshold=2):
    """
    Generate a specular reflection map by subtracting the diffuse image from the RGB image using PyTorch.
    
    :param rgb_image: RGB image as a PIL Image
    :param diffuse_image: Diffuse image as a PIL Image
    :param kernel_size: Size of the box kernel for local smoothing
    :param threshold: Threshold for specular values and normalization
    :return: specular reflection map as a PIL Image
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and convert images to PyTorch tensors
    to_tensor = transforms.ToTensor()
    rgb_tensor = to_tensor(rgb_image).to(device)
    diffuse_tensor = to_tensor(diffuse_image).to(device)

    # Compute specular tensor
    specular_tensor = rgb_tensor - diffuse_tensor

    # Clip negative values to 0
    specular_tensor = torch.clamp(specular_tensor, min=0.0)

    # Apply local smoothing
    padding = kernel_size // 2
    specular_smoothed = F.avg_pool2d(specular_tensor, kernel_size, stride=1, padding=padding)

    # Normalize using the threshold
    specular_normalized = torch.clamp(specular_smoothed / threshold, min=0.0, max=1.0)

    # Convert to grayscale
    specular_gray = specular_normalized.mean(dim=0, keepdim=True)

    # Convert to 0-255 range
    specular_uint8 = (specular_gray * 255).permute(1, 2, 0)
    
    # Convert to PIL Image
    specular_image = Image.fromarray(specular_uint8.squeeze().cpu().numpy().astype(np.uint8))

    return specular_image

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
            delight_image = delight_predictor(input_image, splits_vertical=1, splits_horizontal=1) 
        delight_image.save(output_delight_path)
    else:
        delight_image = Image.open(output_delight_path)
    
    output_mask_path = os.path.join(output_delight_dir, f"{image_name}_mask.png")
    specular_image = generate_specular(input_image, delight_image)
    specular_image.save(output_mask_path)
    
    object_mask_path = os.path.join(input_mask_dir, f"{image_name}.png")
    if os.path.exists(object_mask_path):
        object_mask = Image.open(object_mask_path)
    else:
        object_mask = None
    
    # Calculate the mean reflection score for the image
    mean_score = classify_reflection(specular_image, object_mask)
    print(mean_score)
    return mean_score
        
@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
@click.option('--images', '-i', default="images", help='Path to the images dir')
@click.option('--masks', '-m', default="mask", help='Path to the masks dir')
@click.option('--num_workers', '-w', default=1, help='Number of worker threads')
@click.option('--threshold', '-t', default=0.1, help='Reflection classification threshold')
def main(source_path: str, images: str, masks: str, num_workers: int, threshold: float) -> None:
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    delight_predictor = torch.hub.load("Stable-X/StableDelight", "StableDelight_turbo", 
                                       trust_repo=True, local_cache_dir='/workspace/code/InverseRendering/StableDiffuse/weights')
    
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