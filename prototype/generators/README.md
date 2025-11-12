# Dataset Generators

Two versions of synthetic character instance segmentation dataset generators.

## 📦 Versions

### Generator v5: `generator_v5_mask_based.py`
Uses **binary mask overlap detection** for collision checking.

**Features:**
- Mask-based collision detection using numpy
- Pre and post-rotation overlap checks
- Automatic spacing adjustment (0 to MAX_EXTRA_SPACING)
- Straight + curved text placement
- Fast performance

**Best for:**
- Quick prototyping
- When speed is more important than precision

### Generator v6: `generator_v6_shapely.py` ⭐ RECOMMENDED
Uses **Shapely polygon intersection** for collision checking.

**Features:**
- All features from v5, plus:
- Geometric polygon overlap detection
- Global polygon tracking across entire image
- More accurate collision prevention
- Font character support validation

**Best for:**
- Production datasets
- When accuracy is critical
- Research publications

## 🎯 Key Differences

| Feature | v5 | v6 |
|---------|----|----|
| Collision Method | Binary mask | Shapely polygons |
| Accuracy | Good | Excellent |
| Speed | Faster | Slightly slower |
| Character Overlap | Word-level only | Global tracking |
| Font Validation | No | Yes |

## 🚀 Quick Start

### Basic Usage

```bash
# Using v6 (recommended)
python generator_v6_shapely.py

# Using v5 (faster)
python generator_v5_mask_based.py
```

### Configuration

Edit the config section at the top of each script:

```python
# Output directories
OUTPUT_DIR = "synthetic_dataset"
FONT_DIR = "fonts"

# Image settings
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_IMAGES = 100

# Character placement
MIN_CHARS_PER_IMAGE = 100
MIN_FONT_SIZE = 28
MAX_FONT_SIZE = 56

# Behavior
CURVED_WORD_PROB = 0.3
SAME_COLOR_WORD_PROB = 0.5
USE_PICSUM = True  # Random backgrounds from Picsum
CONTRAST_THRESHOLD = 3.0
```

## 📤 Output Structure

```
OUTPUT_DIR/
├── images/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── annotations/
    ├── 00001.csv
    ├── 00002.csv
    └── ...
```

### CSV Format

Each annotation CSV contains:
```csv
char,polygon_json
A,"[x1,y1,x2,y2,x3,y3,...]"
B,"[x1,y1,x2,y2,x3,y3,...]"
```

## ⚙️ Requirements

```bash
pip install Pillow opencv-python numpy faker requests shapely
```

Note: `shapely` is only required for v6.

## 🎨 Features

### Background Options
- **Picsum Photos**: Random real-world images (requires internet)
- **Solid White**: Fallback when Picsum unavailable

### Text Placement Modes
- **Straight**: Rotated words (-40° to +40°)
- **Curved**: Arc-based text along circular path

### Collision Prevention
- Pre-rotation character overlap check
- Post-rotation character overlap check
- Background mask tracking
- Polygon-based validation (v6 only)

### Color Management
- Automatic contrast calculation
- Configurable contrast threshold
- Same-color or multi-color words

## 📊 Performance

Typical generation speeds (on GPU):
- **v5**: ~10-15 images/minute
- **v6**: ~8-12 images/minute

## 🐛 Troubleshooting

### No fonts found
```
Error: No fonts found in 'fonts/' directory
```
→ Run font download script first

### Low character count
If images have fewer characters than MIN_CHARS_PER_IMAGE:
- Increase MAX_PLACEMENT_ATTEMPTS
- Decrease MIN_CHARS_PER_IMAGE
- Increase MAX_EXTRA_SPACING

### Picsum timeout
If background fetch fails:
- Set `USE_PICSUM = False` for solid backgrounds
- Check internet connection
