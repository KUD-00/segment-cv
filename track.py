import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Track objects in a video using a YOLO model.')
parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
parser.add_argument('source_video', type=str, help='Path to the source video file.')
args = parser.parse_args()

model = YOLO(args.model_path)

results = model.track(source=args.source_video, conf=0.01, tracker="bytetrack.yaml", save=True)

print(results[0].boxes)

id_to_centers = {}

with open("result_list.txt", "w") as file:
    for result in results:
        for box in result.boxes:
            box_id = box.id.item()
            print("--------------------------------")
            print(box.xywh)
            print(box.xywh.shape)

            x_center, y_center = box.xywh[0][0].item(), box.xywh[0][1].item()

            if box_id in id_to_centers:
                id_to_centers[box_id].append((x_center, y_center))
            else:
                id_to_centers[box_id] = [(x_center, y_center)]

    for box_id, centers in id_to_centers.items():
        file.write(f"ID: {box_id}, Centers: {centers}\n")
