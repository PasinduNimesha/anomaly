import os
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage
import xml.etree.ElementTree as ET
from xml.dom import minidom


def run_cluster():
    datasets = ["sixray"]
    num_clusters = [4, 3, 3, 4]

    # Colors defined in MATLAB code (RGB)
    colors = {
        "sixray":  np.array([214, 255, 230]),
        "gdxray":  np.array([255, 221, 255]),
        "compass": np.array([0, 255, 0]),
        "opixray": np.array([0, 255, 0])
    }

    target_colors_override = {
        "compass": np.array([165, 255, 230]),
        "opixray": np.array([198, 216, 255])
    }

    for d_idx, d_name in enumerate(datasets):
        print(f"Processing dataset: {d_name}")
        
        # Paths
        base_path = os.path.join('datasets', d_name, 'results')
        pn_fake = os.path.join(base_path, 'fake')
        pn_real = os.path.join(base_path, 'real')
        pn_results = os.path.join(base_path, 'results')
        pn_annotations = os.path.join(base_path, 'annotations') # New folder for XMLs

        # Create output directories
        os.makedirs(pn_results, exist_ok=True)
        os.makedirs(pn_annotations, exist_ok=True)

        # Get images
        search_path = os.path.join(pn_real, '*.png')
        image_files = glob.glob(search_path)
        
        num_regions = num_clusters[d_idx]
        
        for file_path in image_files:
            fn = os.path.basename(file_path)
            
            path_real = os.path.join(pn_real, fn)
            path_fake = os.path.join(pn_fake, fn)
            
            if not os.path.exists(path_fake):
                continue

            real_img = cv2.imread(path_real)
            fake_img = cv2.imread(path_fake)
            
            if real_img is None or fake_img is None:
                continue

            # Calculate Disparity: dis = 255*(real-fake)
            diff = cv2.subtract(real_img, fake_img) 
            diff_scaled = cv2.multiply(diff, 255) 
            dis = cv2.subtract(np.array([255], dtype=np.uint8), diff_scaled) 
            
            dis_proc = dis.copy()
            
            # --- Dataset Specific Pre-processing ---
            if d_name == "compass" or d_name == "opixray":
                t_crop = 300
                if dis_proc.shape[1] > 2*t_crop:
                    dis_proc[:, 0:t_crop, :] = 0
                    dis_proc[:, -t_crop:, :] = 0
                
                t_thresh = 10
                curr_color = colors[d_name] # RGB
                dis_rgb = cv2.cvtColor(dis_proc, cv2.COLOR_BGR2RGB)
                
                delta = dis_rgb.astype(float) - curr_color.astype(float)
                dist_map = np.sqrt(np.sum(delta**2, axis=2))
                
                mask_high_dist = dist_map < t_thresh
                dis_proc[mask_high_dist] = [255, 255, 255]

            # --- K-Means Clustering ---
            h, w, c = dis_proc.shape
            pixel_data = dis_proc.reshape((-1, 3))
            
            # Note: For speed in production, you might reduce n_init or sample pixels
            kmeans = KMeans(n_clusters=num_regions, random_state=42, n_init=5)
            labels_flat = kmeans.fit_predict(pixel_data.astype(float))
            centers = kmeans.cluster_centers_
            
            # Determine Anomaly Index
            target_co = colors[d_name]
            if d_name in target_colors_override:
                target_co = target_colors_override[d_name]
                
            min_dist = float('inf')
            anomaly_idx = -1
            
            for l_idx in range(len(centers)):
                center = centers[l_idx] 
                center_rgb = center[::-1] # BGR to RGB
                
                dist = np.linalg.norm(center_rgb - target_co)
                if dist < min_dist:
                    min_dist = dist
                    anomaly_idx = l_idx
            
            mask = (labels_flat == anomaly_idx).reshape((h, w)).astype(np.uint8)
            
            # --- Dataset Specific Morphological Operations ---
            kernel_def = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            disk_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            disk_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

            if d_name == "sixray":
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_def, iterations=3)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=10000).astype(np.uint8)
                
                # Column masking
                mask2 = np.zeros((h, w), dtype=np.uint8)
                t_strip = 10
                for i in range(224, w, 224):
                    start = max(0, i - t_strip)
                    end = min(w, i + t_strip)
                    mask2[:, start:end] = 1
                
                mask3 = cv2.bitwise_and(mask, mask2)
                mask = cv2.subtract(mask, mask3)
                
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=10000).astype(np.uint8)
                mask = cv2.dilate(mask, kernel_def, iterations=2)
                mask = cv2.dilate(mask, disk_5, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_5, iterations=2)
                mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=40000).astype(np.uint8)

            elif d_name == "gdxray":
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=10000).astype(np.uint8)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=20000).astype(np.uint8)
                mask = cv2.dilate(mask, kernel_def, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_5)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_5)
                mask = cv2.dilate(mask, kernel_def, iterations=3)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_5)
                mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=40000).astype(np.uint8)

            elif d_name == "compass":
                mask = cv2.dilate(mask, kernel_def, iterations=3)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_5, iterations=2)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=15000).astype(np.uint8)
                mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

            elif d_name == "opixray":
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=15000).astype(np.uint8)
                mask = cv2.erode(mask, kernel_def, iterations=1)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_3)
                mask = cv2.dilate(mask, kernel_def, iterations=3)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_5, iterations=2)
                mask = morphology.remove_small_objects(mask.astype(bool), min_size=5000).astype(np.uint8)
                mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

            # --- Visualization and Annotation Saving ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output_img = real_img.copy()
            detected_boxes = []

            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                detected_boxes.append((x, y, x + w_box, y + h_box))
                
                # Draw on image
                cv2.rectangle(output_img, (x, y), (x+w_box, y+h_box), (0, 255, 255), 10)
                cv2.putText(output_img, 'Anomaly', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
                
                # Highlight in Red channel
                b, g, r = cv2.split(output_img)
                r[mask == 1] = 255
                output_img = cv2.merge([b, g, r])

            # 1. Save Montage Image
            montage = np.hstack((real_img, output_img))
            save_img_path = os.path.join(pn_results, fn)
            cv2.imwrite(save_img_path, montage)

            # Save predicted bounding boxes in XML format
            annotation = ET.Element("annotation")

            folder = ET.SubElement(annotation, "folder")
            folder.text = "X_ray"

            filename = ET.SubElement(annotation, "filename")
            filename.text = fn

            source = ET.SubElement(annotation, "source")
            database = ET.SubElement(source, "database")
            database.text = "The X_ray Database"
            annotation_source = ET.SubElement(source, "annotation")
            annotation_source.text = "The X_ray Database"
            image = ET.SubElement(source, "image")
            image.text = "X_ray"
            flickrid = ET.SubElement(source, "flickrid")
            flickrid.text = "0"

            owner = ET.SubElement(annotation, "owner")
            flickrid_owner = ET.SubElement(owner, "flickrid")
            flickrid_owner.text = "miaocaijing16@mails.ucas.ac.ac"
            name = ET.SubElement(owner, "name")
            name.text = "MeioJane"

            size = ET.SubElement(annotation, "size")
            width = ET.SubElement(size, "width")
            width.text = str(real_img.shape[1])
            height = ET.SubElement(size, "height")
            height.text = str(real_img.shape[0])
            depth = ET.SubElement(size, "depth")
            depth.text = str(real_img.shape[2])

            segmented = ET.SubElement(annotation, "segmented")
            segmented.text = "0"

            # Ensure bounding boxes are scaled to original image dimensions
            original_height, original_width = real_img.shape[:2]
            processed_height, processed_width = dis_proc.shape[:2]

            scale_x = original_width / processed_width
            scale_y = original_height / processed_height

            scaled_boxes = []
            for box in detected_boxes:
                x_min, y_min, x_max, y_max = box
                scaled_boxes.append((
                    int(x_min * scale_x),
                    int(y_min * scale_y),
                    int(x_max * scale_x),
                    int(y_max * scale_y)
                ))

            # Replace detected_boxes with scaled_boxes for saving in XML
            detected_boxes = scaled_boxes

            for box in detected_boxes:
                obj = ET.SubElement(annotation, "object")
                name = ET.SubElement(obj, "name")
                name.text = "Knife"
                pose = ET.SubElement(obj, "pose")
                pose.text = "Unspecified"
                truncated = ET.SubElement(obj, "truncated")
                truncated.text = "0"
                difficult = ET.SubElement(obj, "difficult")
                difficult.text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(box[0])
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(box[1])
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(box[2])
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(box[3])

            tree = ET.ElementTree(annotation)
            xml_save_path = os.path.join(pn_annotations, fn.replace('.png', '.xml'))
            tree.write(xml_save_path)
            


if __name__ == "__main__":
    run_cluster()