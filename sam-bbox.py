import cv2
import numpy as np
import sys
import os
import argparse
import json
from segment_anything import sam_model_registry, SamPredictor
import random

def random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def apply_mask(image, mask):
    color = random_color()
    for i in range(3):
        image[:, :, i] = np.where(mask == 1, image[:, :, i] * 0.4 + color[i] * 0.6, image[:, :, i])
    return image

def draw_masks_on_blank(blank_image, mask):
    color = random_color()
    return apply_mask(blank_image, mask)

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

    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    bboxes = read_bboxes(args.file_path, target_id)

    first_frame_with_all_masks = first_frame.copy()
    blank_with_all_masks = np.zeros((first_frame.shape[0], first_frame.shape[1], 3), dtype=np.uint8)

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
            for m in masks:
                mask_single = (m > 0).astype(np.uint8)
                first_frame_with_all_masks = apply_mask(first_frame_with_all_masks, mask_single)
                blank_with_all_masks = draw_masks_on_blank(blank_with_all_masks, mask_single)

        counter += 1

    cv2.imwrite(os.path.join(video_dir, 'original_with_masks.png'), first_frame_with_all_masks)
    cv2.imwrite(os.path.join(video_dir, 'only_masks.png'), blank_with_all_masks)

    mask_image_path = os.path.join(video_dir, 'only_masks.png')
    image = cv2.imread(mask_image_path)
    if image is None:
        print(f"Failed to read image from {mask_image_path}")
        sys.exit(1)

    # 裁剪图像
    clipped_image = clip_image(image, clicks)
    clipped_image_path = os.path.join(video_dir, f"{os.path.splitext(os.path.basename(args.video_path))[0]}-clipped.png")
    cv2.imwrite(clipped_image_path, clipped_image)

    # 创建分段数组
    segmentation_array = np.zeros((100, 100), dtype=int)
    scaled_image = cv2.resize(clipped_image, (100, 100), interpolation=cv2.INTER_NEAREST)
    segmentation_array[(scaled_image[:, :, 0] != 0) | (scaled_image[:, :, 1] != 0) | (scaled_image[:, :, 2] != 0)] = 1
    
    # 保存分段数组到文本文件
    array_output_filename = os.path.join(video_dir, f'{os.path.splitext(os.path.basename(args.video_path))[0]}-segmentation-array.txt')
    np.savetxt(array_output_filename, segmentation_array, fmt='%d')