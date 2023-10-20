import cv2

# 视频文件路径
video_path = '/home/dl-station/qi/segmentation/DJI_0247-005.MP4'

# 初始化变量
frame_counter = 0
image_counter = 0
success = True

# 创建视频捕获对象
video = cv2.VideoCapture(video_path)

# 获取视频的FPS（每秒帧数）
fps = video.get(cv2.CAP_PROP_FPS)
# fps = 60
# 计算每3秒的帧数
frame_rate = int(fps * 3)

while success and image_counter < 600:
    success, image = video.read()
    
    if frame_counter % frame_rate == 0 and success:
        cv2.imwrite(f'frame{image_counter}.jpg', image)
        image_counter += 1
    
    frame_counter += 1

# 释放视频捕获对象
video.release()