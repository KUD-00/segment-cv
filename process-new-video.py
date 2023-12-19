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

# 使用示例
process_video('/media/dl-station/disk2/qi/空撮/281.mp4', '281.mp4')
