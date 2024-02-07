import cv2
import numpy as np
import os
import argparse
import json
from segment_anything import sam_model_registry, SamPredictor
import random

def random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def apply_mask(image, mask):
    color = random_color() + [100]  # RGBA
    overlay = image.copy()
    overlay[mask > 0] = color
    new_image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    return new_image

def draw_masks_on_blank(image_shape, mask, color):
    blank_image = np.ones(image_shape, dtype=np.uint8) * 255  # white background
    return apply_mask(blank_image, mask, color)

def read_bboxes(file_name, target_id):
    bboxes = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith(f"ID: {target_id}"):
                start = line.find('[')
                end = line.find(']')
                bbox_str = line[start+1:end]
                bbox = np.array([float(x.strip()) for x in bbox_str.split(',')], dtype=np.int32).reshape((-1, 1, 2))
                bboxes.append(bbox)
    return bboxes

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

    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    bboxes = read_bboxes(args.file_path, target_id)
    segmentation_array = np.zeros((100, 100), dtype=int)

    counter = 0
    while success:
        success, frame = video.read()
        if not success or counter >= len(bboxes):
            break

        if counter % 20 == 0:
            box = bboxes[counter]
            masks, scores, logits = predictor.predict(frame, box)
            for mask in masks:
                first_frame_with_mask = apply_mask(first_frame.copy(), mask_cv2, color)
                mask_on_blank = draw_masks_on_blank(first_frame.shape, mask_cv2, color)

        counter += 1

    cv2.imwrite(os.path.join(video_dir, 'original_with_masks.png'), first_frame_with_mask)
    cv2.imwrite(os.path.join(video_dir, 'only_masks.png'), mask_on_blank)