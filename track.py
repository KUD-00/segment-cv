import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Track objects in a video using a YOLO model.')
parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
parser.add_argument('source_video', type=str, help='Path to the source video file.')
args = parser.parse_args()

model = YOLO(args.model_path)

results = model.track(source=args.source_video, show=True, conf=0.1, tracker="bytetrack.yaml", save=True)

print(results[0].boxes)

id_to_centers = {}

with open("result_list.txt", "w") as file:
    for result in results:
        for box in result.boxes:
            # 提取 ID。使用 item() 来从张量中获取标量值
            box_id = box.id.item()  # 假设每个 box 的 id 是单个元素张量

            # 从 xywh 张量中提取中心坐标
            x_center, y_center = box.xywh[0].item(), box.xywh[1].item()

            # 如果这个 ID 已经在字典中，添加新的中心位置
            if box_id in id_to_centers:
                id_to_centers[box_id].append((x_center, y_center))
            else:
                # 否则，为这个 ID 创建一个新的中心位置列表
                id_to_centers[box_id] = [(x_center, y_center)]

    # 将字典内容写入文件
    for box_id, centers in id_to_centers.items():
        file.write(f"ID: {box_id}, Centers: {centers}\n")
