import argparse
import os
from ultralytics import YOLO

# 解析命令行参数
parser = argparse.ArgumentParser(description='Track objects in a video using a YOLO model.')
parser.add_argument('model_path', type=str, help='Path to the YOLO model file.')
parser.add_argument('source_video', type=str, help='Path to the source video file.')
args = parser.parse_args()

# 初始化并使用 YOLO 模型
model = YOLO(args.model_path)
results = model.track(source=args.source_video, conf=0.2, show=True, tracker="bytetrack.yaml", save=True)
print(results[0].boxes)

# 从视频路径中提取视频名称
video_name = os.path.splitext(os.path.basename(args.source_video))[0]

# 构造输出文件名
bboxes_filename = f"{video_name}-bboxes.txt"

# 写入边界框数据
with open(bboxes_filename, "w") as bboxes_file:
    for result in results:
        for box in result.boxes:
            box_id = box.id.item()
            bbox = box.xyxy[0].tolist()
            bboxes_file.write(f"ID: {box_id}, BBox: {bbox}\n")

# 读取用户输入
target_id = input("Please enter the target ID: ")
direction = input("Please enter the direction: ")

# 构造新的文件名
new_bboxes_filename = f"{video_name}-{target_id}-{direction}-bboxes.txt"

# 重命名边界框文件
os.rename(bboxes_filename, new_bboxes_filename)

print(f"Renamed bbox file to {new_bboxes_filename}")