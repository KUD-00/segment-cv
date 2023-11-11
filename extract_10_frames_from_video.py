import sys
import os
from moviepy.editor import VideoFileClip
import imageio

def extract_images(video_path, timestamp, output_dir):
    time_h, time_m, time_s = map(int, timestamp.split(':'))
    time_in_seconds = time_h * 3600 + time_m * 60 + time_s
    start_time = max(0, time_in_seconds - 2)  # Ensure start_time is non-negative
    end_time = time_in_seconds + 2

    with VideoFileClip(video_path) as video:
        duration = video.duration
        end_time = min(duration, end_time)  # Ensure end_time does not exceed video duration
        step = (end_time - start_time) / 4  # Calculate step to get 10 images
        
        for i in range(4):
            time = start_time + i * step
            frame = video.get_frame(time)
            image_path = os.path.join(output_dir, f"{int(time)}.png")
            imageio.imwrite(image_path, frame)
            print(f"Image saved at {image_path}")

if __name__ == "__main__":
    video_path = sys.argv[1]
    timestamp = sys.argv[2]
    output_dir = sys.argv[3]
    extract_images(video_path, timestamp, output_dir)
