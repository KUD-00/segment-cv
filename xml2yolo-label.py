import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os
import chardet  # You need to install this package if you haven't; it's used for detecting character encoding

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Dictionary for label mapping
    labels_map = {
        'car': '10',
        'truck': '9',
        'bus': '8',
        'feright car': '7',
        'feright_car': '5',
        'van': '6',
        # Add more if there are other labels
    }

    annotations = []

    for member in root.findall('object'):
        label = member.find('name').text  # Changed from member[0].text to more accurately target the <name> element
        if label in labels_map:
            label_id = labels_map[label]

            # Check for both polygon and bndbox
            polygon = member.find('polygon')
            bndbox = member.find('bndbox')

            # If it's a polygon
            if polygon is not None:
                points = []
                for pt in polygon:
                    points.append([int(pt.text)])
                contour = np.array(points).reshape((-1, 1, 2))
                x, y, w, h = cv2.boundingRect(contour)

            # If it's a bounding box
            elif bndbox is not None:
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin

            else:
                print(f"Warning: No polygon or bounding box found for label '{label}' in file {xml_file}.")
                continue  # Skip this object and move to the next

            # Convert to YOLO format
            x_center_norm = (x + w / 2) / width
            y_center_norm = (y + h / 2) / height
            width_norm = w / width
            height_norm = h / height

            annotations.append((label_id, x_center_norm, y_center_norm, width_norm, height_norm))
        else:
            print(f"Warning: Label '{label}' not found in labels_map.")

    return annotations
def write_annotations_to_txt(annotations, txt_filepath):
    with open(txt_filepath, 'w') as file:
        for annotation in annotations:
            line = ' '.join(str(x) for x in annotation)
            file.write(line + '\n')

def convert_xml_directory_to_yolo(xml_directory, yolo_directory):
    # Check if yolo directory exists, if not create it
    if not os.path.exists(yolo_directory):
        os.makedirs(yolo_directory)

    for xml_file in os.listdir(xml_directory):
        if xml_file.endswith('.xml') and not xml_file.startswith('._'):  # Skip '._' files
            try:
                xml_filepath = os.path.join(xml_directory, xml_file)

                # Use chardet to detect the character encoding of the file
                rawdata = open(xml_filepath, "rb").read()
                result = chardet.detect(rawdata)
                charenc = result['encoding']

                with open(xml_filepath, 'r', encoding=charenc) as file:  # Use the detected encoding
                    first_line = file.readline()

                annotations = parse_xml(xml_filepath)

                # Construct txt file name
                txt_filename = os.path.splitext(xml_file)[0] + '.txt'
                txt_filepath = os.path.join(yolo_directory, txt_filename)

                write_annotations_to_txt(annotations, txt_filepath)
            except ET.ParseError as e:
                print(f"ParseError: {e} in file {xml_file}")
                continue  # Skip this file if there is an error
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError: {e} in file {xml_file}")
                continue  # Skip this file if there is an error
            except Exception as e:
                print(f"An unexpected error occurred with file {xml_file}: {e}")
                continue  # Skip this file if there is an error


# Specify the directories
xml_directory = '/media/dl-station/disk2/qi/dv-dataset/val/label'
yolo_directory = '/media/dl-station/disk2/qi/dv-dataset/val/trainlabel'

convert_xml_directory_to_yolo(xml_directory, yolo_directory)
