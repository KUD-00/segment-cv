import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import argparse
import random

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back the coordinates to the original size
        scaled_x, scaled_y = int(x / scale_percent * 100), int(y / scale_percent * 100)
        clicks.append((scaled_x, scaled_y))
        if len(clicks) == 4:
            cv2.destroyAllWindows()

def random_color():
    return np.array([random.randint(0, 255) / 255 for _ in range(3)] + [0.6])

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6]), alpha=0.6):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=alpha)

def read_bboxes(file_name, target_id):
    bboxes = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith(f"ID: {target_id}"):
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    bbox_str = line[start+1:end]
                    bbox = [float(x.strip()) for x in bbox_str.split(',')]
                    bboxes.append(bbox)
    return bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract bounding boxes and show masks with random colors.')
    parser.add_argument('file_path', type=str, help='Path to the file containing bounding boxes.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('object_id', type=float, nargs='?', default=None, help='ID of the object to extract bounding boxes for.')
    args = parser.parse_args()

    scale_percent = 50
    clicks = []

    video = cv2.VideoCapture(args.video_path)
    success, first_frame = video.read()
    if not success:
        print("Error reading video.")
        sys.exit(1)
    print("hello")

    resized_first_frame = resize_image(first_frame, scale_percent)
    cv2.imshow('First Frame - Click 4 Points', resized_first_frame)
    cv2.setMouseCallback('First Frame - Click 4 Points', click_event)
    cv2.waitKey(0)

    print("hello")

    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # Adjust for three types of masks.
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    # Placeholder for white-base image with masks for the third subplot.
    white_base = np.ones_like(first_frame) * 255

    bboxes = read_bboxes(args.file_path, args.object_id)
    if bboxes:
        bbox = bboxes[0]
        length = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))

    segmentation_array = np.zeros((100, 100), dtype=int)
    counter = 0
    while success:
        success, frame = video.read()
        if not success or counter >= len(bboxes):
            break

        if counter % 20 == 0:
            box = bboxes[counter]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            np_box = np.array(box)
            masks, scores, logits = predictor.predict(box=np_box, multimask_output=False)
            for mask in masks:
                color = random_color()
                show_mask(mask, axs[0], np.array([0, 0, 1, 1]), 1)  # Blue mask with no transparency.
                show_mask(mask, axs[1], color, 1)  # Colorful mask with no transparency.
                # Apply mask to the white_base for third subplot.
                mask_resized = cv2.resize(mask.astype(np.uint8), (first_frame.shape[1], first_frame.shape[0]))
                white_base[mask_resized > 0] = np.array(color[:3] * 255, dtype=np.uint8)

        counter += 1

    # After processing all masks, show the white_base image with applied masks in the third subplot.
    axs[2].imshow(white_base)

    # Save the original and colorful masks into one image.
    plt.savefig(f'{args.file_path}-segmentation-and-colorful.png', dpi=300)

    # If needed, separate saving logic for individual images can be added here.
