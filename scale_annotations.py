import os
import xml.etree.ElementTree as ET

def rescale_annotations(directory, target_width, target_height):
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Get original size
            size = root.find("size")
            original_width = int(size.find("width").text)
            original_height = int(size.find("height").text)

            # Update size to target dimensions
            size.find("width").text = str(target_width)
            size.find("height").text = str(target_height)

            # Rescale bounding boxes
            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                bndbox.find("xmin").text = str(round(xmin * target_width / original_width, 3))
                bndbox.find("ymin").text = str(round(ymin * target_height / original_height, 3))
                bndbox.find("xmax").text = str(round(xmax * target_width / original_width, 3))
                bndbox.find("ymax").text = str(round(ymax * target_height / original_height, 3))

            # Save the updated XML
            tree.write(file_path)

# Directory containing the XML files
annotations_dir = r"C:\Users\User\Documents\GitHub\anomaly\datasets\sixray\annotations"
rescale_annotations(annotations_dir, 2240, 2240)
