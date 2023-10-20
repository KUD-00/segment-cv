from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('./pretrained_models/yolov8x.pt')  # load a custom model

# Predict with the model
results = model("./split_frames/top_right.jpg", show=True, save=True)
results = model("./split_frames/down_right.jpg", show=True, save=True)
results = model("./split_frames/down_left.jpg", show=True, save=True)


# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    print(masks)
    keypoints = result.keypoints  # Keypoints object for pose outputs
    print(keypoints)
    probs = result.probs  # Class probabilities for classification outputs
    print(probs)
