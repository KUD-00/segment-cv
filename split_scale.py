from PIL import Image

# Load the original image
image = Image.open("frame3.jpg")

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
top_left.save("./split_frames/top_left.jpg")
top_right.save("./split_frames/top_right.jpg")
down_left.save("./split_frames/down_left.jpg")
down_right.save("./split_frames/down_right.jpg")