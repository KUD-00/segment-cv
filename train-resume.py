
import argparse
from ultralytics import YOLO

def train_model(model_path):
    # Load a pretrained YOLO model
    model = YOLO(model_path)

    # Train the model with the specified dataset and number of epochs
    results = model.train(resume=True)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("model", type=str, help="Path to the pretrained model file.")

    args = parser.parse_args()
    
    # Train the model with the provided arguments
    train_model(args.model)
