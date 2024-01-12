import argparse
import cv2

def draw_dots_on_frame(frame, centers, color=(0, 0, 255), radius=5):
    for center in centers:
        cv2.circle(frame, center, radius, color, -1)
    return frame

def read_centers_from_file(file_path, tracking_id):
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(f"ID: {tracking_id},"):
                centers = eval(line.split("Centers: ")[1])
                centers = [(int(x), int(y)) for x, y in centers]
                break
    return centers

def main():
    parser = argparse.ArgumentParser(description='Track a specific object in a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('txt_path', type=str, help='Path to the text file containing tracking data.')
    parser.add_argument('id', type=str, help='ID of the object to track.')
    args = parser.parse_args()

    centers = read_centers_from_file(args.txt_path, args.id)

    if not centers:
        print(f"No data found for ID {args.id}.")
        return

    video = cv2.VideoCapture(args.video_path)
    success, frame = video.read()
    if not success:
        print("Error reading video.")
        return

    frame_with_dots = draw_dots_on_frame(frame, centers)

    cv2.imwrite('tracked.jpg', frame_with_dots)
    print("Image saved as 'tracked.jpg'.")

    video.release()

if __name__ == "__main__":
    main()
