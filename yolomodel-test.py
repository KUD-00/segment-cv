from ultralytics import YOLO
import time

# Load the trained model
model = YOLO("runs/detect/train7/weights/best.pt")

results = model("frame3.jpg", save=True, show_labels=False)

# for i in range(600):
#     model.predict(f'original-frames/frame{i}.jpg', save_txt=True, conf=0.7)