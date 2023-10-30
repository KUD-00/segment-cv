import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything_hq import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import supervision as sv
from ultralytics import YOLO

IMAGE_PATH = "./split_frames/top_right.jpg"

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
    model = YOLO("runs/detect/train13/weights/best.pt")
    results = model("split_frames/top_right.jpg")

    sam_checkpoint = "./pretrained_models/sam_hq_vit_h.pth"
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
    type_9_boxes = bboxes_numpy[bboxes_numpy[:, -1] == 9, :4]

    for i, box in enumerate(type_9_boxes):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_bgr)
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True
        )
        for j, mask in enumerate(masks):
            show_mask(mask, plt.gca())
            show_box(box, plt.gca())
            plt.axis('off')
            plt.savefig(f'output_mask_{i}_{j}.png', dpi=300)
