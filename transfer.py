from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('runs/detect/train7/weights/best.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='transfer.yaml', epochs=10)