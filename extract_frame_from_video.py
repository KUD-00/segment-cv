import cv2
import sys
import os

def extract_frame(video_path, time_str, output_directory):
    # 将时间字符串转换为秒
    minutes, seconds = map(int, time_str.split(':'))
    target_time = minutes * 60 + seconds

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算要提取的帧的编号
    frame_no = int(target_time * fps)

    # 设置视频位置到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Couldn't extract frame at {time_str}.")
        return

    output_file = f"frame_{time_str.replace(':', '-')}.jpg"
    output_path = os.path.join(output_directory, output_file)

    # 保存帧为图片
    cv2.imwrite(output_path, frame)

    # 释放资源
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_frame.py <video_path> <time_str> <output_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    time_str = sys.argv[2]
    output_path = sys.argv[3]

    extract_frame(video_path, time_str, output_path)
