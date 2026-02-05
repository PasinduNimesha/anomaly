import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_bounding_boxes(xml_file):
    """Parse bounding boxes from an XML file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

def draw_bounding_boxes(image_path, bboxes, title, color="green"):
    """Draw bounding boxes on an image."""
    image = Image.open(image_path)
    plt.imshow(image)
    ax = plt.gca()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
    plt.title(title)
    plt.axis("off")

def view_ground_truth_and_predictions(image_path, ground_truth_xml, predicted_xml):
    """View ground truth and predicted bounding boxes side by side."""
    ground_truth_bboxes = parse_bounding_boxes(ground_truth_xml)
    predicted_bboxes = parse_bounding_boxes(predicted_xml)

    plt.figure(figsize=(12, 6))

    # Ground truth bounding boxes
    plt.subplot(1, 2, 1)
    draw_bounding_boxes(image_path, ground_truth_bboxes, title="Ground Truth", color="green")

    # Predicted bounding boxes
    plt.subplot(1, 2, 2)
    draw_bounding_boxes(image_path, predicted_bboxes, title="Predictions", color="red")

    plt.tight_layout()
    plt.show()

# Paths to the folders
image_folder = "C:\\Users\\User\\Documents\\GitHub\\anomaly\\datasets\\sixray\\results\\real"
ground_truth_folder = "C:\\Users\\User\\Documents\\GitHub\\anomaly\\datasets\\sixray\\annotations"
predicted_folder = "C:\\Users\\User\\Documents\\GitHub\\anomaly\\datasets\\sixray\\results\\annotations"

# Iterate through images and display bounding boxes
for image_file in os.listdir(image_folder):
    if image_file.endswith(".png"):  # Assuming images are in PNG format
        image_path = os.path.join(image_folder, image_file)
        ground_truth_xml = os.path.join(ground_truth_folder, image_file.replace(".png", ".xml"))
        predicted_xml = os.path.join(predicted_folder, image_file.replace(".png", ".xml"))

        if os.path.exists(ground_truth_xml) and os.path.exists(predicted_xml):
            print(f"Displaying bounding boxes for {image_file}")
            view_ground_truth_and_predictions(image_path, ground_truth_xml, predicted_xml)
        else:
            print(f"Missing XML file for {image_file}")