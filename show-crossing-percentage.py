import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
import cv2
import imageio

def resize_image(image, scale_percent=50):
    width, height = image.size
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    return image.resize((new_width, new_height))

def on_click(event):
    global clicks, root
    scaled_x, scaled_y = int(event.x), int(event.y)  # No need to rescale back
    clicks.append((scaled_x, scaled_y))
    if len(clicks) == 4:
        root.destroy()

import cv2
import numpy as np

def apply_segmentation_overlay(image, segmentation_array, clicks, scale_percent):
    # 归一化分割数据
    normalized_array = segmentation_array.astype(np.float32) / np.max(segmentation_array)

    # 根据缩放比例调整点击坐标到原始图像的尺寸
    adjusted_clicks = [(int(x * 100 / scale_percent), int(y * 100 / scale_percent)) for x, y in clicks]

    # 计算点击定义的矩形区域
    top_left = min(adjusted_clicks, key=lambda x: x[0]+x[1])
    bottom_right = max(adjusted_clicks, key=lambda x: x[0]+x[1])
    region_width = bottom_right[0] - top_left[0]
    region_height = bottom_right[1] - top_left[1]

    # 计算每个小方格的大小
    cell_width = region_width // 100
    cell_height = region_height // 100

    # 根据normalized_array的值动态计算颜色
    for i in range(100):
        for j in range(100):
            value = normalized_array[j, i]
            if value > 0:  # 只有当值大于0时才绘制方块
                red_intensity = int(255 * value)
                color = (0, 0, red_intensity)  # BGR格式
                start_x = int(top_left[0] + cell_width * i)
                start_y = int(top_left[1] + cell_height * j)
                cv2.rectangle(image, (start_x, start_y),
                              (start_x + cell_width, start_y + cell_height),
                              color, cv2.FILLED)

    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Overlay segmentation on video frame.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('array_path', type=str, help='Path to the segmentation array file.')
    args = parser.parse_args()

    scale_percent = 50
    clicks = []

    # Load the first frame and display it
    reader = imageio.get_reader(args.video_path)
    first_frame = reader.get_next_data()
    image = Image.fromarray(first_frame)
    resized_image = resize_image(image, scale_percent)

    root = tk.Tk()
    tk_image = ImageTk.PhotoImage(resized_image)
    panel = tk.Label(root, image=tk_image)
    panel.pack(side="bottom", fill="both", expand="yes")
    panel.bind("<Button-1>", on_click)
    root.mainloop()
    print(clicks)

    # Load segmentation array
    segmentation_array = np.loadtxt(args.array_path, dtype=int)
    print(segmentation_array)

    # Convert PIL image to OpenCV image format
    first_frame_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply segmentation overlay
    result_image = apply_segmentation_overlay(first_frame, segmentation_array, clicks, scale_percent)

    # Save the result image
    output_path = args.video_path.rsplit('.', 1)[0] + '_segmentation_overlay.png'
    cv2.imwrite(output_path, result_image)
