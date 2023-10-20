from PIL import Image

# Load the split images
top_left = Image.open("./split_segment_frames/top_left.jpg")
top_right = Image.open("./split_segment_frames/top_right.jpg")
down_left = Image.open("./split_segment_frames/down_left.jpg")
down_right = Image.open("./split_segment_frames/down_right.jpg")

# Get the size of the split images
width, height = top_left.size

# Create a new image with the original size
image = Image.new("RGB", (width*2, height*2))

# Paste the split images into the new image
image.paste(top_left, (0, 0))
image.paste(top_right, (width, 0))
image.paste(down_left, (0, height))
image.paste(down_right, (width, height))

# Resize the image to the original size
image = image.resize((width, height), Image.ANTIALIAS)

# Save the merged image
image.save("./merged_frames/merged_image.jpg")