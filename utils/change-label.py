import os

def transform_file_content(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if not parts:  # skip empty lines
                continue
            if parts[0] == "0":
                parts[0] = "10"
            elif parts[0] == "1":
                parts[0] = "9"
            file.write(' '.join(parts) + '\n')

def process_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.txt'):
                transform_file_content(os.path.join(root, file))

if __name__ == '__main__':
    dir_path = "/media/dl-station/disk2/qi/v3/train/labels"
    process_directory(dir_path)
