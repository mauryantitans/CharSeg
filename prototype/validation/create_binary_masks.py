"""
Binary Mask Visualizer - Variant 1
Converts polygon annotations to white-on-black binary masks for quality inspection.

This creates pure mask images showing character coverage areas.
"""
import os
import json
import csv
import numpy as np
import cv2

# Configuration
ANNOTATIONS_FOLDER = "/content/synthetic_dataset_full/annotations"  # Change this path
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

def polygons_to_mask(polygons, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Convert list of polygon coordinates to binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1,2))
        cv2.fillPoly(mask, [pts], 255)
    return mask

def create_masks_from_annotations(anno_folder):
    """Process all CSV annotations and create mask images"""
    masks_dir = os.path.join(anno_folder, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(anno_folder) if f.endswith(".csv")]
    
    print(f"Found {len(csv_files)} annotation files")
    
    for csv_file in csv_files:
        csv_path = os.path.join(anno_folder, csv_file)
        polygons = []
        
        # Read polygons from CSV
        with open(csv_path, newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                poly_json = row["polygon_json"]
                poly = json.loads(poly_json)
                polygons.append(poly)
        
        # Create mask
        mask = polygons_to_mask(polygons)
        
        # Save mask
        mask_path = os.path.join(masks_dir, csv_file.replace(".csv", "_mask.png"))
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask: {mask_path}")

if __name__ == "__main__":
    create_masks_from_annotations(ANNOTATIONS_FOLDER)
    print("\n✅ All masks created successfully!")
