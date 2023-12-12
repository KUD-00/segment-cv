from moviepy.editor import VideoFileClip
import os
import sys

def extract_video_segment(input_video_path, start_time, end_time, output_dir):
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    output_video_name = f"{name}-{start_time}-{end_time}{ext}"
    output_video_path = os.path.join(output_dir, output_video_name)

    with VideoFileClip(input_video_path) as video:
        video_segment = video.subclip(start_time, end_time)
        video_segment.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: video path, start time, end time, output directory")
    else:
        input_video = sys.argv[1]
        start_time = sys.argv[2]
        end_time = sys.argv[3]
        output_dir = sys.argv[4]

        extract_video_segment(input_video, start_time, end_time, output_dir)
