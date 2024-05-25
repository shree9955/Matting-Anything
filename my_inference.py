# https://github.com/ultralytics/ultralytics/issues/11781 , this is to resolve status: CUDNN_STATUS_NOT_SUPPORTED

import os
import random
import cv2
from scipy import ndimage

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision

import networks
import utils

# Grounding DINO
import sys
sys.path.insert(0, './GroundingDINO')
from groundingdino.util.inference import Model

# SAM
sys.path.insert(0, './segment-anything')
from segment_anything.utils.transforms import ResizeLongestSide

# SD
from diffusers import StableDiffusionPipeline


torch.backends.cudnn.benchmark = True

transform = ResizeLongestSide(1024)
PALETTE_back = (51, 255, 146)

GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"
mam_checkpoint = "checkpoints/mam_vitb.pth"
output_dir = "outputs"
device = "cuda"
background_list = os.listdir('assets/backgrounds')

# initialize MAM
mam_model = networks.get_generator_m2m(seg='sam_vit_b', m2m='sam_decoder_deep')
mam_model.to(device)
checkpoint = torch.load(mam_checkpoint, map_location=device)
mam_model.m2m.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
mam_model = mam_model.eval()

# initialize GroundingDINO
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=device)

# initialize StableDiffusionPipeline
generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
generator.to(device)

def run_grounded_sam(input_image, text_prompt, task_type, background_prompt, background_type, box_threshold, text_threshold, iou_threshold, scribble_mode, guidance_mode):
    
    global groundingdino_model, mam_model, generator

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    # load image
    image_ori = input_image["image"]
    scribble = input_image["mask"]
    original_size = image_ori.shape[:2]

    if task_type == 'text':
        if text_prompt is None:
            raise ValueError('Please input non-empty text prompt')
        with torch.no_grad():
            detections, phrases = grounding_dino_model.predict_with_caption(
                image=cv2.cvtColor(image_ori, cv2.COLOR_RGB2BGR),
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

        if len(detections.xyxy) > 1:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                iou_threshold,
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
    
        bbox = detections.xyxy[np.argmax(detections.confidence)]
        bbox = transform.apply_boxes(bbox, original_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(device)

    image = transform.apply_image(image_ori)
    image = torch.as_tensor(image).to(device)
    image = image.permute(2, 0, 1).contiguous()

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(device)

    image = (image - pixel_mean) / pixel_std

    h, w = image.shape[-2:]
    pad_size = image.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    image = F.pad(image, (0, padw, 0, padh))

    if task_type == 'scribble_point':
        scribble = scribble.transpose(2, 1, 0)[0]

        labeled_array, num_features = ndimage.label(scribble >= 255)

        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
        centers = np.array(centers)
        centers = transform.apply_coords(centers, original_size)
        point_coords = torch.from_numpy(centers).to(device)
        point_coords = point_coords.unsqueeze(0).to(device)
        point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)
        if scribble_mode == 'split':
            point_coords = point_coords.permute(1, 0, 2)
            point_labels = point_labels.permute(1, 0)
            
        sample = {'image': image.unsqueeze(0), 'point': point_coords, 'label': point_labels, 'ori_shape': original_size, 'pad_shape': pad_size}
    
    elif task_type == 'scribble_box':
        scribble = scribble.transpose(2, 1, 0)[0]
        labeled_array, num_features = ndimage.label(scribble >= 255)
        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
        centers = np.array(centers)
        x_min = centers[:, 0].min()
        x_max = centers[:, 0].max()
        y_min = centers[:, 1].min()
        y_max = centers[:, 1].max()
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox = transform.apply_boxes(bbox, original_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).to(device)

        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    elif task_type == 'text':
        sample = {'image': image.unsqueeze(0), 'bbox': bbox.unsqueeze(0), 'ori_shape': original_size, 'pad_shape': pad_size}
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    with torch.no_grad():
        feas, pred, post_mask = mam_model.forward_inference(sample)

        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os4 = alpha_pred_os4[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]
        alpha_pred_os1 = alpha_pred_os1[..., : sample['pad_shape'][0], : sample['pad_shape'][1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, sample['ori_shape'], mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, sample['ori_shape'], mode="bilinear", align_corners=False)
        
        if guidance_mode == 'mask':
            weight_os8 = utils.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=10, train_mode=False)
            post_mask[weight_os8 > 0] = alpha_pred_os8[weight_os8 > 0]
            alpha_pred = post_mask.clone().detach()
        else:
            weight_os8 = utils.get_unknown_box_from_mask(post_mask)
            alpha_pred_os8[weight_os8 > 0] = post_mask[weight_os8 > 0]
            alpha_pred = alpha_pred_os8.clone().detach()

        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
        alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
        alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]
       
        alpha_pred = alpha_pred[0][0].cpu().numpy()

    alpha_rgb = cv2.cvtColor(np.uint8(alpha_pred * 255), cv2.COLOR_GRAY2RGB)
    if background_type == 'random':
        background_path = os.path.join('assets/backgrounds', random.choice(background_list))
        background_img = cv2.imread(background_path)
        background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
        com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
        com_img = np.uint8(com_img)
    else:
        if background_prompt is None:
            raise ValueError('Please input non-empty background prompt')
        else:
            background_img = generator(background_prompt).images[0]
            background_img = np.array(background_img)
            background_img = cv2.resize(background_img, (image_ori.shape[1], image_ori.shape[0]))
            com_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.uint8(background_img)
            com_img = np.uint8(com_img)

    green_img = alpha_pred[..., None] * image_ori + (1 - alpha_pred[..., None]) * np.array([PALETTE_back], dtype='uint8')
    green_img = np.uint8(green_img)

    # Save the results instead of showing them
    # cv2.imwrite(os.path.join(output_dir, 'composite_with_background.png'), cv2.cvtColor(com_img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(output_dir, 'green_screen.png'), cv2.cvtColor(green_img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(os.path.join(output_dir, 'alpha_matte.png'), cv2.cvtColor(alpha_rgb, cv2.COLOR_RGB2BGR))

    return [(com_img, 'composite_with_background'), (green_img, 'green_screen'), (alpha_rgb, 'alpha_matte')]

import glob
# Define the path to the directory containing images
image_directory = '/home/aniket/shree/Matting-Anything/13-07-2022/13_07_2022/Ground_RGB_Photos/Healthy'

# Get a list of all image files in the directory
image_files = glob.glob(os.path.join(image_directory,'*.jpg'))  

# Process each image in the directory
for image in image_files:
    # Read and process the image
    our_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    our_mask = np.zeros((our_image.shape[0], our_image.shape[1], 1), dtype=np.uint8)

    # Simulate inputs
    input_image = {"image": our_image, "mask": our_mask}
    text_prompt = "the tree in the middle"
    task_type = "text"
    background_prompt = "downtown area in New York"
    background_type = "generated_by_text"
    box_threshold = 0.25
    text_threshold = 0.25
    iou_threshold = 0.5
    scribble_mode = "split"
    guidance_mode = "alpha"

    # Call the function
    results = run_grounded_sam(input_image, text_prompt, task_type, background_prompt, background_type, box_threshold, text_threshold, iou_threshold, scribble_mode, guidance_mode)
    # Process the results as needed
    print(f"Processed results for image: {image}")

    # Loop through the results and save each image
    for result_image, label in results:
        # Construct the output file path
        output_file_name = f"{os.path.splitext(os.path.basename(image))[0]}_{label}.png"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # Save the output image
        cv2.imwrite(output_file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))


