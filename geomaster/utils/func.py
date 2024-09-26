import cv2
import numpy as np
import torch


def load_and_split_image(image_path):

    img = cv2.imread(image_path)  
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img[:, :, [0, 2]] = img[:, :, [2, 0]]
    # print(img.shape)
    mask = np.ones(img.shape[:2], dtype=bool)  

    
    for c in range(img.shape[2]):
        mask &= (img[:, :, c] > 0)

    
    # img[mask, 2] = 255 - img[mask, 2]
   
    height, width = img.shape[:2]
    print(height)
    print(width)
    
    if width == 512 and height == 512:
        
        crop_positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
    elif width == 1024 and height == 256:
        
        crop_positions = [(0, 0), (256, 0), (512, 0), (768, 0)]
    elif width == 768 and height == 512:
        
        crop_positions = [
            (0, 0), (256, 0), (512, 0),  
            (0, 256), (256, 256), (512, 256)  
        ]
    else:
        print(height)
        print(width)
        raise ValueError("image size should be 512x512, 1024x256 or 768x512¡£")

    
    imgs = [img[y:y+256, x:x+256] for x, y in crop_positions]

    
    images_data = []
    for sub_img in imgs:
        
        # sub_img = cv2.flip(sub_img, 1)

        # [0, 1]
        img_array = sub_img / 255.0
        images_data.append(img_array)

    
    if width == 512 or width == 1024:
        images_tensor = np.stack(images_data, axis=0)  
    elif width == 768:
        images_tensor = np.stack(images_data, axis=0)  # (6, 256, 256, 3)

    return images_tensor