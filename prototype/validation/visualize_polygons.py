"""
Green Line Polygon Visualizer - Variant 2
Draws polygon boundaries as green lines over original dataset images.

This allows detailed inspection of polygon accuracy against actual characters.
"""
import os
import csv
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
IMAGES_DIR = "/content/synthetic_dataset_full/images"  # Change these paths
ANNOTATIONS_DIR = "/content/synthetic_dataset_full/annotations"
OUTPUT_DIR = "/content/synthetic_dataset_full/visualized_polygons"

# Create output folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_polygons_on_image(image_path, annotation_path, output_path):
    """Draw polygon boundaries on image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    # Read annotations and draw polygons
    with open(annotation_path, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poly = json.loads(row["polygon_json"])
            pts = np.array(poly, dtype=np.int32).reshape((-1,2))
            # Draw green polyline (closed shape)
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)

    # Save with Matplotlib (ensures correct color space)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_dataset(images_dir, annotations_dir, output_dir):
    """Process all images in dataset"""
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    
    print(f"Found {len(image_files)} images to process")
    
    for filename in image_files:
        img_path = os.path.join(images_dir, filename)
        ann_path = os.path.join(annotations_dir, filename.replace(".png", ".csv"))
        out_path = os.path.join(output_dir, filename)

        if not os.path.exists(ann_path):
            print(f"⚠ No annotation found for {filename}, skipping.")
            continue

        draw_polygons_on_image(img_path, ann_path, out_path)
        print(f"✅ Saved visualization: {out_path}")

if __name__ == "__main__":
    # Run for the whole dataset
    process_dataset(IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR)
    print(f"\n✅ All visualizations saved in: {OUTPUT_DIR}")
