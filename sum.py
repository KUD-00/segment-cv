import os
import numpy as np
import argparse

def find_and_sum_arrays(root_dir, pattern='-segmentation-array.txt'):
    sum_array = np.zeros((100, 100), dtype=int)

    # 遍历root_dir下的所有文件和目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(pattern):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing {file_path}...")
                # 读取并累加数组
                array = np.loadtxt(file_path, dtype=int)
                sum_array += array

    return sum_array

def save_array(array, output_path):
    np.savetxt(output_path, array, fmt='%d')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and sum all *-segmentation-array.txt files under a given directory.')
    parser.add_argument('directory', type=str, help='The directory to search in.')
    args = parser.parse_args()

    # 执行查找和累加操作
    summed_array = find_and_sum_arrays(args.directory)
    output_path = os.path.join(args.directory, 'sum.txt')
    # 保存最终累加的数组
    save_array(summed_array, output_path)
    print(f"Summed array saved to {output_path}")
