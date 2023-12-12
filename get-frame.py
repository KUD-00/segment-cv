import cv2
import sys
import os

def extract_frame(video_path, time_str, output_directory):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # 获取视频的总时长（秒）
    total_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

    # 解析时间字符串
    if '/' in time_str:
        # 如果时间字符串是比例形式（例如 1/2）
        numerator, denominator = map(int, time_str.split('/'))
        target_time = total_duration * numerator / denominator
    else:
        # 如果时间字符串是分钟:秒形式
        minutes, seconds = map(int, time_str.split(':'))
        target_time = minutes * 60 + seconds

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(target_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Couldn't extract frame at {time_str}.")
        return

    # 提取视频文件的基本名称
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]

    # 构造输出文件名
    output_file = f"{video_base_name}-{time_str.replace(':', '-').replace('/', '-')}.jpg"
    output_path = os.path.join(output_directory, output_file)

    cv2.imwrite(output_path, frame)
    cap.release()

def is_video_file(file_name):
    video_file_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(file_name.endswith(ext) for ext in video_file_extensions)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_frame.py <video_path>/<video_folder_path> <time_str> <output_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    time_str = sys.argv[2]
    output_path = sys.argv[3]

    if os.path.isdir(video_path):
        for file_name in os.listdir(video_path):
            full_path = os.path.join(video_path, file_name)
            if os.path.isfile(full_path) and is_video_file(file_name):
                extract_frame(full_path, time_str, output_path)
    else:
        extract_frame(video_path, time_str, output_path)
