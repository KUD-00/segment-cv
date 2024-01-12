import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
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
    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict()

    for i, mask in enumerate(masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_bgr)
        show_mask(mask, plt.gca())
        plt.axis('off')
        plt.savefig(f'output_mask_{i}.png', dpi=300)
