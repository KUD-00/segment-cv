import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Track objects in a video using a YOLO model.')
parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
parser.add_argument('source_video', type=str, help='Path to the source video file.')
args = parser.parse_args()

model = YOLO(args.model_path)

results = model.track(source=args.source_video, conf=0.2, show=True, tracker="bytetrack.yaml", save=True)

print(results[0].boxes)

id_to_centers = {}
bboxes_data = []

with open("centers.txt", "w") as centers_file, open("bboxes.txt", "w") as bboxes_file:
    for result in results:
        for box in result.boxes:
            box_id = box.id.item()

            x_center, y_center = box.xywh[0][0].item(), box.xywh[0][1].item()
            bbox = box.xyxy[0].tolist()

            if box_id in id_to_centers:
                id_to_centers[box_id].append((x_center, y_center))
            else:
                id_to_centers[box_id] = [(x_center, y_center)]

            bboxes_data.append((box_id, bbox))

    for box_id, centers in id_to_centers.items():
        centers_file.write(f"ID: {box_id}, Centers: {centers}\n")

    for box_id, bbox in bboxes_data:
        bboxes_file.write(f"ID: {box_id}, BBox: {bbox}\n")

