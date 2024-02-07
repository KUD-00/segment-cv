#!/bin/bash

# 使用命令行参数作为输入目录
INPUT_DIR="$1" # 使用第一个命令行参数作为输入目录

# 检查是否提供了输入目录
if [ -z "$INPUT_DIR" ]; then
  echo "Usage: $0 /path/to/input/directory"
  exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: The specified directory does not exist."
  exit 1
fi

# 遍历目录中的所有.mp4文件
for video in "$INPUT_DIR"/*.mp4; do
  # 检查是否有文件匹配，如果没有，则跳过迭代
  [[ -e "$video" ]] || continue

  # 提取视频文件的名称（不包含扩展名）
  video_name=$(basename "${video}" .mp4)

  # 创建与视频名称相同的目录（如果不存在的话）
  mkdir -p "$INPUT_DIR/$video_name"

  # 将视频移动到新创建的目录中
  mv "$video" "$INPUT_DIR/$video_name/"
done

echo "All MP4 files have been moved."
