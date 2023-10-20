import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import supervision as sv



if __name__ == "__main__":
    sam_checkpoint = "./pretrained_models/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for i in range(8):
        print("image:   ",i)
        # hq_token_only: False means use hq output to correct SAM output. 
        #                True means use hq output only. 
        #                Default: False
        hq_token_only = False 
        # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
        # For images contain single object, we suggest to set hq_token_only = True
        # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False
        mask_generator = SamAutomaticMaskGenerator(sam)

        image_bgr = cv2.imread('./original_frames/frame'+str(i)+'.jpg')
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = mask_generator.generate(image_rgb)

        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_bgr, detections)
        cv2.imwrite("./afterSegment_frames/frame"+str(i)+".jpg", annotated_image)  
    



