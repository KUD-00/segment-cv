from PIL import Image
import argparse
import os

def crop_center_and_resize(img_path, output_folder):
    # 打开图片
    with Image.open(img_path) as img:
        # 获取图片的宽和高
        w, h = img.size
        # 计算中心位置
        left = w * 0.25
        top = h * 0.25
        right = w * 0.75
        bottom = h * 0.75
        # 裁剪中心部分
        cropped = img.crop((left, top, right, bottom))
        # 放大两倍
        resized = cropped.resize((int(cropped.width * 2), int(cropped.height * 2)))
        # 创建输出路径
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, base_name)
        # 保存处理后的图片
        resized.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the center of the images in the folder and resize them.")
    parser.add_argument("input_path", help="Path to the input image or folder.")
    parser.add_argument("output_path", help="Path to save the processed images. Should be a folder.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if os.path.isdir(args.input_path):
        for root, dirs, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(root, file)
                    crop_center_and_resize(file_path, args.output_path)
    else:
        crop_center_and_resize(args.input_path, args.output_path)
