import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

def parse_xml(xml_file):
    """
    Parses Pascal VOC XML file and returns a list of bounding boxes.
    Format: [[xmin, ymin, xmax, ymax], ...]
    """
    if not os.path.exists(xml_file):
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    
    return boxes

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_results(gt_folder, pred_folder, iou_threshold=0.5):
    """
    Evaluates predictions against ground truths.
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    # Get all ground truth files
    gt_files = glob.glob(os.path.join(gt_folder, "*.xml"))
    
    print(f"Found {len(gt_files)} Ground Truth files.")

    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_folder, filename)

        gt_boxes = parse_xml(gt_file)
        pred_boxes = parse_xml(pred_file)

        # Keep track of which GT boxes have been matched
        gt_matched = [False] * len(gt_boxes)

        # Check every predicted box
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            # Find the best matching ground truth box for this prediction
            for i, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # Determine if it's a TP or FP
            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1 # Duplicate detection for the same object
            else:
                fp += 1 # No overlap or poor overlap

        # Any GT box not matched is a False Negative
        fn += gt_matched.count(False)

    # Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 30)
    print(f"Evaluation Results (IoU Threshold: {iou_threshold})")
    print("-" * 30)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Path to the folder containing your Ground Truth XMLs (e.g., SIXray annotations)
    GT_FOLDER_PATH = "C:\\Users\\User\\Documents\\GitHub\\anomaly\\datasets\\sixray\\annotations"
    
    # Path to the folder containing your Predicted XMLs (from cluster.py)
    PRED_FOLDER_PATH = "C:\\Users\\User\\Documents\\GitHub\\anomaly\\datasets\\sixray\\results\\annotations"

    if os.path.exists(GT_FOLDER_PATH) and os.path.exists(PRED_FOLDER_PATH):
        evaluate_results(GT_FOLDER_PATH, PRED_FOLDER_PATH, iou_threshold=0.5)
    else:
        print("Error: Please check your folder paths in the script.")