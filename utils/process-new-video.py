import argparse
import os
import glob
from moviepy.editor import VideoFileClip

def process_video(input_path, output_path):
    # 读取视频
    clip = VideoFileClip(input_path)

    # 获取视频的宽度和高度
    width, height = clip.size

    # 首先竖直方向上分成两半，取左半部分
    left_half = clip.crop(x1=0, y1=0, x2=width/2, y2=height)

    # 计算左半部分的新宽度和高度
    new_width, new_height = left_half.size

    # 然后在水平方向上将左半部分分成四份，取中间两份
    middle_section = left_half.crop(x1=0, y1=new_height/4, x2=new_width, y2=new_width/4*3)

    # 放大两倍
    final_clip = middle_section.resize(2.0)

    # 输出视频
    final_clip.write_videofile(output_path, codec='libx264')

def process_directory(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理目录下的所有视频文件
    for input_path in glob.glob(os.path.join(input_dir, "*.mp4")):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        print(f"处理视频：{input_path}")
        process_video(input_path, output_path)
        print(f"视频处理完成，输出到：{output_path}")

def main():
    parser = argparse.ArgumentParser(description="处理视频文件或目录下所有视频文件")
    parser.add_argument("input_path", help="输入视频文件的路径或包含视频文件的目录路径")
    parser.add_argument("output_path", help="输出视频文件的路径或目录路径")

    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        process_directory(args.input_path, args.output_path)
    else:
        process_video(args.input_path, args.output_path)

if __name__ == "__main__":
    main()