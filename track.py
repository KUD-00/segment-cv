import argparse
import os
import json
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Track objects in a video using a YOLO model.')
parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
parser.add_argument('source_video', type=str, help='Path to the source video file.')
args = parser.parse_args()

model = YOLO(args.model_path)
results = model.track(source=args.source_video, conf=0.2, show=True, tracker="bytetrack.yaml", save=True)

video_dir = os.path.dirname(args.source_video)
video_name = os.path.splitext(os.path.basename(args.source_video))[0]

bboxes_filename = os.path.join(video_dir, "bboxes.txt")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

with open(bboxes_filename, "w") as bboxes_file:
    for result in results:
        for box in result.boxes:
            box_id = box.id.item()
            bbox = box.xyxy[0].tolist()
            bboxes_file.write(f"ID: {box_id}, BBox: {bbox}\n")

target_id = input("Please enter the target ID: ")
start_point = input("Please enter the start point: ")
end_point = input("Please input end point: ")
data = {
    "target_id": target_id,
    "start_point": start_point,
    "end_point": end_point
}

json_filename = os.path.join(video_dir, f"{video_name}.json")
with open(json_filename, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"Data saved to {json_filename}")
