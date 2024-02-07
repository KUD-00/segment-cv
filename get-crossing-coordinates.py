import argparse
import os
import json
import tkinter as tk
from PIL import Image, ImageTk
import imageio

def resize_image(image, scale_percent=50):
    width, height = image.size
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    return image.resize((new_width, new_height))

def on_click(event):
    global clicks, root
    scaled_x, scaled_y = int(event.x * 100 / scale_percent), int(event.y * 100 / scale_percent)
    clicks.append((scaled_x, scaled_y))
    if len(clicks) == 4:
        root.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    scale_percent = 50
    clicks = []

    reader = imageio.get_reader(args.video_path)
    first_frame = reader.get_next_data()
    image = Image.fromarray(first_frame)

    resized_image = resize_image(image, scale_percent)

    root = tk.Tk()
    tk_image = ImageTk.PhotoImage(resized_image)
    panel = tk.Label(root, image=tk_image)
    panel.pack(side="bottom", fill="both", expand="yes")
    panel.bind("<Button-1>", on_click)

    root.mainloop()

    video_dir = os.path.dirname(args.video_path)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    json_filename = os.path.join(video_dir, f"{video_name}.json")

    if os.path.exists(json_filename):
        with open(json_filename, "r+") as f:
            data = json.load(f)
            data["clicks"] = clicks
            f.seek(0)
            json.dump(data, f, indent=4)
    else:
        with open(json_filename, "w") as f:
            data = {"clicks": clicks}
            json.dump(data, f, indent=4)

    print(f"Clicks saved to {json_filename}")
