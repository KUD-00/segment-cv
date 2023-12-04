import cv2
import numpy as np
# Example usage
input_video = '/media/dl-station/disk2/qi/img-proc/crossing.mp4'  # Replace with your video file path
output_video = './crossing-1.mp4'  # The output file path

def process_video(input_file, output_file):
    # Open the input video
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frames_to_process = int(60 * fps) # 1 minute of frames

    while frames_to_process > 0:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to twice its size
        resized_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)

        # Extract the center of the resized frame
        start_x = resized_frame.shape[1] // 4
        start_y = resized_frame.shape[0] // 4
        center_frame = resized_frame[start_y:start_y + height, start_x:start_x + width]

        # Write the frame
        out.write(center_frame)

        frames_to_process -= 1

    # Release everything if job is finished
    cap.release()
    out.release()

# Example usage
process_video(input_video, output_video)
