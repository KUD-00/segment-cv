from PIL import Image
import argparse

def split_and_scale_image(input_image_path, output_directory):
    # Load the original image
    image = Image.open(input_image_path)

    # Get the width and height of the image
    width, height = image.size

    # Split the image into four parts
    top_left = image.crop((0, 0, width/2, height/2))
    top_right = image.crop((width/2, 0, width, height/2))
    down_left = image.crop((0, height/2, width/2, height))
    down_right = image.crop((width/2, height/2, width, height))

    # Scale each part to the original image's size
    top_left = top_left.resize((width, height), Image.ANTIALIAS)
    top_right = top_right.resize((width, height), Image.ANTIALIAS)
    down_left = down_left.resize((width, height), Image.ANTIALIAS)
    down_right = down_right.resize((width, height), Image.ANTIALIAS)

    # Save the split and scaled images
    top_left.save(f"{output_directory}/top_left.jpg")
    top_right.save(f"{output_directory}/top_right.jpg")
    down_left.save(f"{output_directory}/down_left.jpg")
    down_right.save(f"{output_directory}/down_right.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split an image into four parts and scale them.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('output_directory', type=str, help='Directory to save the output images')
    args = parser.parse_args()
    split_and_scale_image(args.image_path, args.output_directory)

