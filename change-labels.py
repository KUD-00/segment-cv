import os

def change_class_labels(directory, changes):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            # Read the current annotations
            with open(filepath, "r") as file:
                lines = file.readlines()

            changed_lines = []
            for line in lines:
                elements = line.strip().split(' ')
                class_id = int(elements[0])

                # If the current class_id is in the changes dict, change it
                if class_id in changes:
                    elements[0] = str(changes[class_id])
                
                changed_line = ' '.join(elements)
                changed_lines.append(changed_line)

            # Write the new annotations to the file
            with open(filepath, "w") as file:
                for line in changed_lines:
                    file.write(f"{line}\n")


if __name__ == "__main__":
    # The directory containing your YOLO label files
    directory = "/media/dl-station/disk2/qi/dv-dataset/val/labels"  # please change this to your directory

    # Define the changes you want to make
    changes = {
        5: 9,  # change class 5 to 9
        7: 9,  # change class 7 to 9
        8: 10, # change class 8 to 10
        6: 10  # change class 6 to 10
    }

    change_class_labels(directory, changes)
