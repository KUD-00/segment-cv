import os
from PIL import Image

class_to_id = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14
}

def convert_to_yolo_format(input_file, img_folder, output_file):
    # Extract image filename from annotation filename
    img_filename = os.path.basename(input_file).replace('.txt', '.png')  # Assumes your images are .jpg
    img_path = os.path.join(img_folder, img_filename)

    with Image.open(img_path) as img:
        img_width, img_height = img.size

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            elements = line.strip().split()
            coords = [float(x) for x in elements[:8]]
            class_name = elements[8]
            class_id = class_to_id[class_name]

            min_x = min(coords[0::2])
            max_x = max(coords[0::2])
            min_y = min(coords[1::2])
            max_y = max(coords[1::2])

            x_center = (min_x + max_x) / 2
            y_center = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y

            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            f_out.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Define paths
annotation_dir = "/media/dl-station/disk2/dota-dataset/valset_reclabelTxt"  # Original annotations directory
image_dir = "/media/dl-station/disk2/dota-dataset/images/val"  # Images directory
output_dir = "/media/dl-station/disk2/dota-dataset/yolo-valTxt"  # YOLO formatted annotations directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(annotation_dir):
    if filename.endswith('.txt'):
        input_file_path = os.path.join(annotation_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        convert_to_yolo_format(input_file_path, image_dir, output_file_path)

print("Conversion completed!")