import tkinter as tk
from PIL import Image, ImageTk

def on_click(event):
    # 打印点击的坐标
    print('Clicked at:', event.x, event.y)

# 创建Tkinter的根窗口
root = tk.Tk()
root.title('Click on Image')

# 加载图片（请替换为您的图片路径）
image_path = '/media/dl-station/disk2/qi/shiomi-videos/crossing1-alter-scaled/256_1/only-mask.png'
image = Image.open(image_path)

# 将PIL图像转换为Tkinter可用的格式
tk_image = ImageTk.PhotoImage(image)

# 创建一个标签来显示图片，并绑定点击事件
label = tk.Label(root, image=tk_image)
label.pack()
label.bind('<Button-1>', on_click)

# 运行Tkinter事件循环
root.mainloop()
