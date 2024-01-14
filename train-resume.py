import argparse
from ultralytics import YOLO

def train_model(model_path):
    model = YOLO(model_path)

    results = model.train(resume=True)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("model", type=str, help="Path to the pretrained model file.")

    args = parser.parse_args()
    
    train_model(args.model)
