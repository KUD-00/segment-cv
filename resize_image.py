from PIL import Image

def resize_image(input_image_path, output_image_path, size=(640, 384)):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    desired_width, desired_height = size

    # 计算等比例缩放的尺寸
    ratio = min(desired_width/width, desired_height/height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # 创建一个新的空白图片，尺寸为384*640，背景颜色为白色
    new_image = Image.new("RGB", (desired_width, desired_height), (255, 255, 255))
    
    # 将原图等比例缩放后贴到新的空白图片上
    resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
    new_image.paste(resized_image, ((desired_width - new_width) // 2, (desired_height - new_height) // 2))

    new_image.save(output_image_path)

# 调用函数
resize_image('input/center_frames/frame_4-59.jpg', 'output.jpg')
