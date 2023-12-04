from ultralytics import YOLO

model = YOLO('runs/detect/train14/weights/best.pt')  # Load a custom trained model

# Perform tracking with the model
results = model.track(source="./output_video.mp4", show=True, conf=0.1, tracker="bytetrack.yaml", save=True)  # Tracking with default tracker

print(results[0].boxes)

# 创建一个空字典来存储 ID 和对应的中心坐标数组
id_to_centers = {}

# 遍历结果集
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
