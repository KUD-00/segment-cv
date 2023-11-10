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
    model.predict(IMAGE_PATH, save=True, imgsz=320, conf=0.2, hide_labels=True)

