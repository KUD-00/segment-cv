import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import supervision as sv
from ultralytics import YOLO
import sys

if len(sys.argv) != 2:
    print("Usage: python script_name.py path_to_image")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

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
    model = YOLO("runs/detect/train18/weights/best.pt")
    results = model(IMAGE_PATH)

    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    bboxes_tensor = torch.tensor(results[0].boxes.data, device='cuda:0')
    bboxes_numpy = bboxes_tensor.cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    for i, box in enumerate(bboxes_numpy):
        confidence = box[4]  # Assuming the confidence score is at index 4
        if confidence > 0.7:
            bbox_coords = box[:4]  # Extract only the bounding box coordinates
            try:
                masks, scores, logits = predictor.predict(
                    box=bbox_coords,  # Pass only the coordinates
                    multimask_output=True
                )
                for j, mask in enumerate(masks):
                    show_mask(mask, plt.gca(), random_color=True)
                    show_box(bbox_coords, plt.gca())
            except Exception as e:
                print(f"Error processing box {i}: {e}")
    
    plt.axis('off')
    plt.savefig('output_all_masks.png', dpi=300)
