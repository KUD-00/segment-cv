import argparse
from ultralytics import YOLO

def train_model(model_path, data_yaml, num_epochs):
    # Load a pretrained YOLO model
    model = YOLO(model_path)

    # Train the model with the specified dataset and number of epochs
    results = model.train(data=data_yaml, epochs=num_epochs)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Script")
    parser.add_argument("model", type=str, help="Path to the pretrained model file.")
    parser.add_argument("data", type=str, help="Path to the YAML file with the dataset configuration.")
    parser.add_argument("epoch", type=int, help="Number of training epochs.")

    args = parser.parse_args()
    
    # Train the model with the provided arguments
    train_model(args.model, args.data, args.epoch)
