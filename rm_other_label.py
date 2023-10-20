import os

def filter_files(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.txt'):
                filepath = os.path.join(foldername, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()

                with open(filepath, 'w') as file:
                    for line in lines:
                        if line.startswith('9 ') or line.startswith('10 '):
                            file.write(line)

# Replace 'your_directory' with the path to the specific folder you want to process
filter_files('/media/dl-station/disk2/dota-dataset/labels/val')
