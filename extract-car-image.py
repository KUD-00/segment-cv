import numpy as np
import torch
import cv2
import os
from ultralytics import YOLO
import argparse
from datetime import datetime

def main(IMAGE_PATH, OUTPUT_PATH, MODEL_PATH, SCALE_FACTOR):
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    image_bgr = cv2.imread(IMAGE_PATH)
    
    bboxes_tensor = torch.tensor(results[0].boxes.data, device='cuda:0')
    bboxes_numpy = bboxes_tensor.cpu().numpy()
    type_9_boxes = bboxes_numpy[bboxes_numpy[:, -1] == 9, :4]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, box in enumerate(type_9_boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped = image_bgr[y1:y2, x1:x2]
        scaled = cv2.resize(cropped, (int(cropped.shape[1]*SCALE_FACTOR), int(cropped.shape[0]*SCALE_FACTOR)), interpolation=cv2.INTER_CUBIC)
        
        filename = f"{os.path.basename(IMAGE_PATH).split('.')[0]}_{current_datetime}_output_scaled_{i}.png"
        output_filepath = os.path.join(OUTPUT_PATH, filename)
        cv2.imwrite(output_filepath, scaled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and scale up image sections based on bboxes.')
    parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
    parser.add_argument('image_path', type=str, help='Path to the input image or directory')
    parser.add_argument('output_path', type=str, help='Path to save the processed images. Should be a directory.')
    parser.add_argument('scale_factor', type=int, help='Scaling factor for image resizing.')
    args = parser.parse_args()

    if os.path.isdir(args.image_path):
        for root, dirs, files in os.walk(args.image_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(root, file)
                    main(file_path, args.output_path, args.model_path, args.scale_factor)
    else:
        main(args.image_path, args.output_path, args.model_path, args.scale_factor)
