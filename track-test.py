import argparse
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

def draw_mask_on_frame(frame, centers, mask, mask_size=(100, 100)):
    for center in centers:
        # 调整掩码大小以适应指定尺寸
        resized_mask = cv2.resize(mask, mask_size, interpolation=cv2.INTER_NEAREST)
        # 计算掩码放置的位置
        x, y = center
        x_start, y_start = x - mask_size[0] // 2, y - mask_size[1] // 2
        x_end, y_end = x_start + mask_size[0], y_start + mask_size[1]

        # 检查边界以避免错误
        if x_start < 0 or y_start < 0 or x_end > frame.shape[1] or y_end > frame.shape[0]:
            continue

        # 应用掩码
        mask_indices = resized_mask > 0
        frame[y_start:y_end, x_start:x_end][mask_indices] = resized_mask[mask_indices]
    return frame

def read_centers_from_file(file_path, tracking_id):
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(f"ID: {tracking_id},"):
                centers = eval(line.split("Centers: ")[1])
                centers = [(int(x), int(y)) for x, y in centers]
                break
    return centers

def extract_mask(image_path):
    # 假设这里是一个从图像中提取掩码的函数
    # 你需要根据你的具体情况来实现这个函数
    # 下面是一个示例函数调用
    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, _, _ = predictor.predict()
    return masks[0]  # 假设我们使用第一个掩码

def main():
    parser = argparse.ArgumentParser(description='Apply a mask to specific locations in a video frame.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('txt_path', type=str, help='Path to the text file containing tracking data.')
    parser.add_argument('mask_image_path', type=str, help='Path to the image file for mask extraction.')
    parser.add_argument('id', type=str, help='ID of the object to track.')
    args = parser.parse_args()

    centers = read_centers_from_file(args.txt_path, args.id)

    if not centers:
        print(f"No data found for ID {args.id}.")
        return

    video = cv2.VideoCapture(args.video_path)
    success, frame = video.read()
    if not success:
        print("Error reading video.")
        return

    mask = extract_mask(args.mask_image_path)
    frame_with_masks = draw_mask_on_frame(frame, centers, mask)

    cv2.imwrite('tracked.jpg', frame_with_masks)
    print("Image saved as 'tracked.jpg'.")

    video.release()

if __name__ == "__main__":
    main()
