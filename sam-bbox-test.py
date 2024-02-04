import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os
import argparse
import random

def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale back the coordinates to the original size
        scaled_x, scaled_y = int(x / scale_percent * 100), int(y / scale_percent * 100)
        clicks.append((scaled_x, scaled_y))
        if len(clicks) == 4:
            cv2.destroyAllWindows()

def random_color():
    return np.array([random.randint(0, 255) / 255 for _ in range(3)] + [0.6])

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6]), alpha=0.6):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=alpha)

def read_bboxes(file_name, target_id):
    bboxes = []
    with open(file_name, 'r') as file:
        for line in file:
            if line.startswith(f"ID: {target_id}"):
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    bbox_str = line[start+1:end]
                    bbox = [float(x.strip()) for x in bbox_str.split(',')]
                    bboxes.append(bbox)
    return bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract bounding boxes and show masks with random colors.')
    parser.add_argument('file_path', type=str, help='Path to the file containing bounding boxes.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('object_id', type=float, nargs='?', default=None, help='ID of the object to extract bounding boxes for.')
    args = parser.parse_args()

    scale_percent = 50  # 初始化全局变量
    clicks = []  # 初始化点击列表

    video = cv2.VideoCapture(args.video_path)
    success, first_frame = video.read()
    if not success:
        print("Error reading video.")
        sys.exit(1)

    resized_first_frame = resize_image(first_frame, scale_percent)
    cv2.imshow('First Frame - Click 4 Points', resized_first_frame)
    cv2.setMouseCallback('First Frame - Click 4 Points', click_event)
    cv2.waitKey(0)
    print("hello")


    # Setup model and device
    sam_checkpoint = "./pretrained_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("hello")

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))  # 三个subplot: 原始掩码、彩色掩码、检测数组
    axs[0].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    axs[1].axis('off')
    # 创建全白底图用于掩码覆盖检测
    white_base = np.ones_like(first_frame) * 255
    axs[2].imshow(white_base)
    axs[2].axis('off')

    bboxes = read_bboxes(args.file_path, args.object_id)
    if bboxes:
        bbox = bboxes[0]
        length = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))

    # 用于存储每个小区域是否被覆盖的二维数组
    segmentation_array = np.zeros((100, 100), dtype=int)

    counter = 0
    while success:
        success, frame = video.read()
        if not success or counter >= len(bboxes):
            break

        if counter % 20 == 0:
            box = bboxes[counter]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            np_box = np.array(box)
            masks, scores, logits = predictor.predict(box=np_box, multimask_output=False)
            for mask in masks:
                color = random_color()  # 为每个掩码分配随机颜色
                show_mask(mask, axs[0], alpha=1)  # 在原始掩码图中显示掩码，无透明度
                show_mask(mask, axs[1], color=color, alpha=1)  # 在彩色掩码图中显示掩码，无透明度
                # 在全白底图上绘制掩码以检测覆盖区域，这里直接使用黑色以简化检测逻辑
                show_mask(mask, axs[2], color=np.array([0, 0, 0, 1]), alpha=1)

        counter += 1

    plt.savefig(f'{args.file_path}-original.png', dpi=300)
    plt.savefig(f'{args.file_path}-colorful.png', dpi=300)

    # 转换 axs[2] 中的图像，以使用 OpenCV 分析
    axs[2].figure.canvas.draw()  # 更新画布
    img = np.frombuffer(axs[2].figure.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(axs[2].figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 分析图像并更新 segmentation_array
    cell_width = img.shape[1] // 100
    cell_height = img.shape[0] // 100
    for i in range(100):
        for j in range(100):
            x_start = i * cell_width
            y_start = j * cell_height
            cell = img[y_start:y_start+cell_height, x_start:x_start+cell_width]
            if np.any(cell != 255):  # 如果单元格内有非白色像素
                segmentation_array[j, i] = 1

    # 保存 segmentation_array 到文本文件
    array_output_filename = f'{args.file_path}-{int(length)}-segmentation-array.txt'
    np.savetxt(array_output_filename, segmentation_array, fmt='%d')
