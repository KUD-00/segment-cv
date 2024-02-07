import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os
import argparse
import random
import json

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

def clip_image(image, points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    return image[y1:y2, x1:x2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract bounding boxes and show masks with random colors.')
    parser.add_argument('file_path', type=str, help='Path to the file containing bounding boxes.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    args = parser.parse_args()
    
    video_dir = os.path.dirname(args.video_path)

    json_filename = os.path.splitext(args.video_path)[0] + ".json"
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)
            clicks = data.get("clicks", [])
            target_id = data.get("target_id")
    else:
        print("JSON file not found.")
        sys.exit(1)

    video = cv2.VideoCapture(args.video_path)
    success, first_frame = video.read()
    if not success:
        print("Error reading video.")
        sys.exit(1)

    # Setup model and device
    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fig1, ax1 = plt.subplots() 
    ax1.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    ax1.axis('off')

    fig2, ax2 = plt.subplots()
    white_base = np.ones_like(first_frame) * 255
    ax2.imshow(white_base)
    ax2.axis('off')

    bboxes = read_bboxes(args.file_path, target_id)
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
                show_mask(mask, ax1, alpha=1)
                show_mask(mask, ax2, color=np.array([0, 0, 0, 1]), alpha=1)

        counter += 1

    fig1.savefig(os.path.join(video_dir, 'original.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    fig2.savefig(os.path.join(video_dir, 'only-mask.png'), dpi=300, bbox_inches='tight', pad_inches=0)

    image = cv2.imread(os.path.join(video_dir, 'only-mask.png'))
    clipped_image = clip_image(image, clicks)
    clipped_image_path = os.path.join(video_dir, f"{os.path.splitext(os.path.basename(args.video_path))[0]}-clipped.png")
    cv2.imwrite(clipped_image_path, clipped_image)
    
    fig3, ax3 = plt.subplots() 
    ax3.imshow(cv2.cvtColor(clipped_image, cv2.COLOR_BGR2RGB))
    ax3.axis('off')

    ax3.figure.canvas.draw()
    img = np.frombuffer(ax3.figure.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(ax3.figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cell_width = img.shape[1] // 100
    cell_height = img.shape[0] // 100
    for i in range(100):
        for j in range(100):
            x_start = i * cell_width
            y_start = j * cell_height
            cell = img[y_start:y_start+cell_height, x_start:x_start+cell_width]
            if np.any(cell != 255):
                segmentation_array[j, i] = 1

    array_output_filename = os.path.join(video_dir, f'{os.path.splitext(os.path.basename(args.video_path))[0]}-segmentation-array.txt')
    np.savetxt(array_output_filename, segmentation_array, fmt='%d')
