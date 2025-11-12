# Background Images for CharSeg

## Overview

This directory should contain 10,000 background images for synthetic text generation.

## Image Requirements

- **Format**: JPG or PNG
- **Size**: At least 1024×768 pixels (will be resized if larger)
- **Variety**: Diverse textures, colors, and complexity levels
- **Content**: Natural scenes, textures, patterns (avoid images with text)

## Recommended Sources

### Option 1: COCO Dataset (Recommended)
Download a subset of MS COCO images:
```bash
# Will be implemented in scripts/download_backgrounds.py
python scripts/download_backgrounds.py --source coco --count 10000
```

### Option 2: Textures.com
- Visit: https://www.textures.com/
- Download texture packs
- Free tier: Limited downloads
- Paid tier: Unlimited access

### Option 3: Unsplash
- Visit: https://unsplash.com/
- Use Unsplash API to download images
- Free tier: 50 requests/hour

### Option 4: Custom Dataset
Place your own images in this directory:
```
datasets/backgrounds/
├── image_0001.jpg
├── image_0002.jpg
├── ...
└── image_10000.jpg
```

## Automatic Download Script

The `scripts/download_backgrounds.py` script will:
1. Download images from specified source
2. Resize to standard dimensions (1024×768)
3. Filter out images with text (optional)
4. Organize in this directory
5. Create metadata file

## Usage

```bash
# Download 10,000 images from COCO
python scripts/download_backgrounds.py --source coco --count 10000

# Download from Unsplash
python scripts/download_backgrounds.py --source unsplash --count 10000 --api_key YOUR_KEY

# Use local images (just place them here and run validation)
python scripts/validate_backgrounds.py
```

## Status

- [ ] Background images downloaded
- [ ] Images validated (size, format)
- [ ] Metadata generated

**Current count**: 0 / 10,000

---

**Note**: Make sure you have sufficient disk space (~5-10 GB) for 10,000 images.
