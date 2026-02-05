import os
import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
from skimage import morphology
from scipy import ndimage
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_pascal_voc_xml(filename, orig_width, orig_height, boxes, dataset_name, proc_width, proc_height):
    """
    Creates a Pascal VOC XML string for the given image and bounding boxes.
    Adjusts bounding box coordinates to match the original image dimensions.
    """
    annotation = ET.Element('annotation')
    
    ET.SubElement(annotation, 'folder').text = dataset_name
    ET.SubElement(annotation, 'filename').text = filename
    
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = dataset_name
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(orig_width)
    ET.SubElement(size, 'height').text = str(orig_height)
    ET.SubElement(size, 'depth').text = '3'
    
    ET.SubElement(annotation, 'segmented').text = '0'
    
    for box in boxes:
        # box format: [x, y, w, h]
        x, y, w, h = box
        
        # Scale bounding box coordinates to original dimensions
        x_min = int(x * (orig_width / proc_width))
        y_min = int(y * (orig_height / proc_height))
        x_max = int((x + w) * (orig_width / proc_width))
        y_max = int((y + h) * (orig_height / proc_height))
        
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'Anomaly'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x_min)
        ET.SubElement(bndbox, 'ymin').text = str(y_min)
        ET.SubElement(bndbox, 'xmax').text = str(x_max)
        ET.SubElement(bndbox, 'ymax').text = str(y_max)
        
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    return xml_str

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
                detected_boxes.append([x, y, w_box, h_box])
                
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

            # 2. Save Annotation (XML)
            if detected_boxes:
                # Construct XML filename (replace extension with .xml)
                xml_fn = os.path.splitext(fn)[0] + '.xml'
                save_xml_path = os.path.join(pn_annotations, xml_fn)
                
                xml_content = create_pascal_voc_xml(fn, w, h, detected_boxes, d_name)
                
                with open(save_xml_path, "w") as f:
                    f.write(xml_content)

if __name__ == "__main__":
    run_cluster()