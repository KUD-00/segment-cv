import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import supervision as sv
from ultralytics import YOLO
import sys
import argparse
import ast

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

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract bounding boxes for a given ID from a file.')
    parser.add_argument('file_path', type=str, help='Path to the file containing bounding boxes.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('object_id', type=float, help='ID of the object to extract bounding boxes for.')
    args = parser.parse_args()

    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    video = cv2.VideoCapture(args.video_path)
    success, first_frame = video.read()
    if not success:
        print("Error reading video.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    counter = 0
    bboxes = read_bboxes(args.file_path, args.object_id)
    while success:
        success, frame = video.read()
        if not success:
            break

        if counter % 20 == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            if counter < len(bboxes):
                box = bboxes[counter]
                print(box)
                np_box = np.array(box)
                masks, scores, logits = predictor.predict(box=np_box, multimask_output=False)
                for mask in masks:
                    show_mask(mask, ax, random_color=True)
                    # show_box(np_box, ax)

        counter += 1

    plt.savefig('combined_masks.png', dpi=300)