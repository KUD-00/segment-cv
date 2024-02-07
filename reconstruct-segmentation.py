import numpy as np
import matplotlib.pyplot as plt
import cv2

# 假设您的数组是从文本文件中读取的
array_path = "256_1-1-4>3-bboxes.txt-segmentation-array.txt" # 替换为您的文件路径
# 读取数组
with open(array_path, 'r') as file:
    array_str = file.read().split()  # 使用 split() 分割空格分隔的字符串
    segmentation_array = np.array([int(num.strip()) for num in array_str]).reshape(100, 100)

# 设置图像的初始大小
img_size = 500  # 假设图像大小为500x500像素
block_size = img_size // 100  # 计算每个块的大小

# 创建一个空白的图像
image = np.ones((img_size, img_size)) * 255  # 创建一个全白的图像

# 根据数组中的值填充图像
for i in range(100):
    for j in range(100):
        if segmentation_array[i, j] == 1:
            # 如果数组中的值为1，则绘制黑色块
            image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0

# 显示图像
plt.imshow(image, cmap='gray')
plt.show()

# 保存图像
cv2.imwrite('reconstructed_image.png', image)
