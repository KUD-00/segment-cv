from moviepy.editor import VideoFileClip

def extract_first_minute(input_video_path, output_video_path):
    with VideoFileClip(input_video_path) as video:
        # Extract the first 60 seconds (first minute)
        first_minute = video.subclip(0, 1)
        # Write the result to the output file
        first_minute.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

# Example usage
input_video = '/home/dl-station/qi/segmentation/crossing-1.mp4'  # Replace with your video file path
output_video = 'crossing-1-original.mp4'  # The output file path
extract_first_minute(input_video, output_video)