import cv2
import numpy as np
import os

def process_images_and_masks(image_dir, mask_dir, output_dir):
    # Get a list of all images and masks
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('_alpha_matte.png')])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_file, mask_file in zip(image_files, mask_files):
        # Read the image and the mask
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale

        # Resize the mask to match the image dimensions if they are different
        if mask.shape[:2] != image.shape[:2]:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure the mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create an output image that is black (all zeros)
        output = np.zeros_like(image)
        
        # Copy the regions of the original image that correspond to the white regions of the mask
        output[binary_mask == 255] = image[binary_mask == 255]
        
        # Optional: create a bounding box around the non-black area in the output for tighter cropping
        coords = cv2.findNonZero(binary_mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image using the bounding box
        cropped_image = output[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Save the cropped image
        output_image_path = os.path.join(output_dir, f'cropped_{image_file}')
        cv2.imwrite(output_image_path, cropped_image)
        
        # Optionally, clear variables to free up memory
        del image, mask, binary_mask, output, cropped_image

# Directories containing images and masks
image_dir = '/home/aniket/shree/Matting-Anything/13-07-2022/13_07_2022/Ground_RGB_Photos/Healthy'
mask_dir = '/home/aniket/shree/Matting-Anything/outputs'
output_dir = '/home/aniket/shree/Matting-Anything/outputs'

# Process all images and masks
process_images_and_masks(image_dir, mask_dir, output_dir)
