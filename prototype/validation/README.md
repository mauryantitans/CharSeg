# Validation Scripts

Visualization tools to verify polygon annotation quality.

## 📦 Variants

Two approaches for validating that polygon annotations accurately match character shapes.

### Variant 1: `create_binary_masks.py` - Binary Mask Visualizer

Creates white-on-black mask images showing character coverage.

**Output:**
- Pure binary masks (white = character, black = background)
- All characters from one image combined
- Saved in `annotations/masks/` folder

**Use cases:**
- Quick visual inspection of character coverage
- Debugging overlap issues
- Verifying polygon fill accuracy

**Example output:**
```
annotations/
├── masks/
│   ├── 00001_mask.png
│   ├── 00002_mask.png
│   └── ...
```

### Variant 2: `visualize_polygons.py` - Green Line Polygon Visualizer ⭐ RECOMMENDED

Draws green polygon contours over original dataset images.

**Output:**
- Original image preserved
- Green polylines showing polygon boundaries
- Saved in `visualized_polygons/` folder

**Use cases:**
- Detailed inspection of polygon accuracy
- Comparing polygon boundaries to actual character pixels
- Quality control for publications/reports

**Example output:**
```
visualized_polygons/
├── 00001.png
├── 00002.png
└── ...
```

## 🚀 Quick Start

### Binary Masks (Variant 1)

```bash
python create_binary_masks.py
```

Edit configuration:
```python
ANNOTATIONS_FOLDER = "/path/to/annotations"  # Folder with CSV files
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
```

### Polygon Overlays (Variant 2)

```bash
python visualize_polygons.py
```

Edit configuration:
```python
IMAGES_DIR = "/path/to/images"              # Original images
ANNOTATIONS_DIR = "/path/to/annotations"    # CSV annotations
OUTPUT_DIR = "/path/to/visualized_polygons" # Output folder
```

## 🎯 When to Use Each

| Use Case | Binary Masks | Polygon Overlays |
|----------|-------------|------------------|
| Quick overview | ✅ | ❌ |
| Detailed inspection | ❌ | ✅ |
| Overlap detection | ✅ | ❌ |
| Pixel-level accuracy check | ❌ | ✅ |
| Publication figures | ❌ | ✅ |
| Debugging fill issues | ✅ | ❌ |

## 📊 Expected Input Format

Both scripts expect annotations in CSV format:

```csv
char,polygon_json
A,"[x1,y1,x2,y2,x3,y3,...]"
B,"[x1,y1,x2,y2,x3,y3,...]"
```

## 🔧 Troubleshooting

### No masks/overlays generated
- Verify annotation folder path is correct
- Check CSV files have correct format
- Ensure polygon_json contains valid coordinates

### Masks look wrong
- Check IMAGE_WIDTH and IMAGE_HEIGHT match your dataset
- Verify polygon coordinates are within image bounds
- Look for negative or out-of-range coordinates

### Missing visualizations
- Check that image files exist in IMAGES_DIR
- Verify image and annotation filenames match
- Ensure file extensions are correct (.png for images, .csv for annotations)

## 🎨 Visualization Details

### Binary Masks
- Uses `cv2.fillPoly()` to fill polygon regions
- White pixels (255) represent characters
- Black pixels (0) represent background
- Good for programmatic analysis

### Polygon Overlays
- Uses `cv2.polylines()` for contour drawing
- Green lines (0, 255, 0) with 2px thickness
- Original image preserved underneath
- Good for human inspection

## 📝 Technical Notes

### Performance
- Binary masks: Very fast (~100 images/second)
- Polygon overlays: Slower due to matplotlib (~5-10 images/second)

### Memory Usage
- Both variants process one image at a time
- Minimal memory footprint
- Safe for large datasets

### Output Quality
- Binary masks: Standard PNG (lossless)
- Polygon overlays: High DPI (150) for print quality

## 💡 Tips

1. **Start with binary masks** for quick sanity check
2. **Use polygon overlays** for detailed quality control
3. **Check random samples** rather than entire dataset
4. **Compare with training results** to verify model is learning correctly
5. **Look for common issues**:
   - Characters cut off at edges
   - Overlapping polygons
   - Gaps in character regions
   - Incorrect topology (holes missing in O, A, etc.)

## 📚 Requirements

```bash
# For binary masks
pip install opencv-python numpy

# For polygon overlays (additional)
pip install matplotlib
```

## 🔗 Related Scripts

- **Data Generation**: `../generators/` (creates the annotations)
- **Training**: `../training/` (uses the annotations)
- **Inference**: `../inference/` (produces similar visualizations)

## 🎯 Quality Checklist

Use these visualizations to verify:
- ✅ Polygons tightly fit character boundaries
- ✅ No overlapping characters
- ✅ Holes in letters (O, P, A, etc.) are preserved
- ✅ Rotation is applied correctly
- ✅ All characters are visible and complete
- ✅ Background is clean (no stray pixels)
