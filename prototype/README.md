# Prototype Code - Character Instance Segmentation

This folder contains the original prototype code from Jupyter notebooks, organized into modular components.

## 📁 Structure

```
prototype/
├── font_download/          # Font acquisition scripts (2 variants)
├── generators/             # Dataset generators v5 & v6
├── validation/             # Visualization tools (2 variants)
├── training/               # Model training code (v1.2)
├── inference/              # Inference script (v1.1)
└── README.md              # This file
```

## 🔄 Workflow

1. **Font Download** → Download Google Fonts (all or curated list)
2. **Data Generation** → Generate synthetic text images with character polygons
3. **Validation** → Visualize polygons to verify quality
4. **Training** → Train Mask R-CNN model
5. **Inference** → Run predictions on new images

## 📝 Version History

### Generators
- **v5**: Mask-based collision detection
- **v6**: Shapely polygon-based collision detection (more accurate)

### Training
- **v1.2**: Industry-standard state dict format, resume/fine-tune modes

### Inference
- **v1.1**: Compatible with v1.2 training output, auto-detects num_classes

## 🚀 Quick Start

Each subdirectory contains:
- Working scripts from the original notebook
- README explaining the code
- Configuration options

See individual README files in each folder for detailed usage.

## ⚠️ Note

This is the **original prototype code** preserved for reference. For production use, refer to the main project structure with modularized components.
