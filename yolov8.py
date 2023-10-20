from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8x.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='dataset.yaml', epochs=40)