import sys
import os
import glob
from PIL import Image

def process_image(input_path, output_path):
    img = Image.open(input_path)

    width, height = img.size

    left_half = img.crop((0, 0, width/2, height))

    new_width, new_height = left_half.size

    middle_section = left_half.crop((0, new_height/4, new_width, new_height/4*3))

    new_width, new_height = middle_section.size

    final_img = middle_section.resize((int(new_width * 2), int(new_height * 2)))

    final_img.save(output_path)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path in glob.glob(input_dir + '/*'):
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            base_name = os.path.basename(file_path)
            output_file_name = os.path.splitext(base_name)[0] + '-resized' + os.path.splitext(base_name)[1]
            output_path = os.path.join(output_dir, output_file_name)

            process_image(file_path, output_path)
            print(f"Processed {file_path} -> {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.isdir(input_path):
        process_directory(input_path, output_path)
    elif os.path.isfile(input_path):
        process_image(input_path, output_path)
    else:
        print("Input path does not exist.")
        sys.exit(1)
