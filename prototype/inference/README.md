# Inference Scripts

Run trained models on new images to predict character instances.

## 📦 Version

### `inference_v1_1.py` - Inference Script v1.1

Compatible with models trained using `train_maskrcnn_v1_2.py`

## 🎯 Features

### Smart Checkpoint Loading
- Automatically detects checkpoint format
- Handles both dict (`{"model_state": ...}`) and direct state_dict
- Auto-infers `num_classes` from model weights
- No manual configuration needed

### Output Types

1. **Binary Mask** (`{image_name}.png`)
   - White characters on black background
   - All detected characters combined
   - Clean mask for visualization or post-processing

2. **Overlay** (`overlay_{image_name}.png`)
   - Original image with annotations
   - Green bounding boxes around each character
   - Yellow contours showing mask boundaries
   - Visual debugging and quality inspection

### Configurable Threshold
- Adjust `SCORE_THRESHOLD` to control detection sensitivity
- Higher = fewer but more confident detections
- Lower = more detections but potentially more false positives

## 🚀 Quick Start

### 1. Configure Paths

Edit the configuration section:

```python
# Path to trained model checkpoint
MODEL_PATH = "/path/to/trained_model.pth"

# Input directory with images to process
INPUT_DIR = "test_images"

# Output directory for results
OUTPUT_DIR = "test_results"

# Confidence threshold (0.0 to 1.0)
SCORE_THRESHOLD = 0.5
```

### 2. Prepare Input Images

Place images in the input directory:
```
test_images/
├── image1.png
├── image2.jpg
└── ...
```

Supported formats: `.png`, `.jpg`, `.jpeg`

### 3. Run Inference

```bash
python inference_v1_1.py
```

### 4. Check Results

```
test_results/
├── image1.png              # Binary mask
├── overlay_image1.png      # Annotated overlay
├── image2.jpg
├── overlay_image2.jpg
└── ...
```

## 📊 Output Examples

### Binary Mask
- Pure white-on-black segmentation masks
- Shows exact detected character regions
- Useful for:
  - Text removal/inpainting
  - Character region analysis
  - Further processing pipelines

### Overlay
- Visual inspection of model predictions
- Shows both detection boxes and mask contours
- Useful for:
  - Quality assessment
  - Debugging false positives/negatives
  - Comparing different model versions

## 🎛️ Adjusting Detection Sensitivity

### Conservative (High Precision)
```python
SCORE_THRESHOLD = 0.7  # Only very confident detections
```
- Fewer detections
- Higher accuracy
- May miss some characters

### Balanced (Default)
```python
SCORE_THRESHOLD = 0.5  # Balanced precision/recall
```
- Good trade-off
- Recommended for most use cases

### Aggressive (High Recall)
```python
SCORE_THRESHOLD = 0.3  # More detections
```
- Catches more characters
- May include false positives
- Better for ensuring no characters are missed

## 🔧 Troubleshooting

### Model fails to load
```
RuntimeError: Could not detect number of classes
```
→ Check that model was trained with v1.2 training script
→ Verify checkpoint file is not corrupted

### No detections on images
- Try lowering `SCORE_THRESHOLD`
- Check if images are similar to training data
- Verify model was trained properly

### Out of memory
- Process images one at a time (already implemented)
- Resize large images before inference
- Use CPU if GPU memory is limited

### Wrong predictions
- Verify model was trained on similar data
- Check if fine-tuning is needed
- Consider retraining with more diverse data

## 📈 Performance

Typical inference speeds:
- **GPU (RTX 3090)**: ~0.1-0.2 seconds per 1024×1024 image
- **GPU (GTX 1080)**: ~0.3-0.5 seconds per image
- **CPU**: ~2-5 seconds per image

Batch processing is done sequentially to minimize memory usage.

## 🎨 Visualization Colors

- **Bounding boxes**: Green (0, 255, 0)
- **Mask contours**: Yellow (0, 255, 255)
- **Binary mask**: White (255) on black (0)

Can be customized in the code if needed.

## 📝 Technical Details

### Model Architecture
- Mask R-CNN with ResNet-50 FPN
- Must match training configuration
- Automatically reconstructed from checkpoint

### Image Processing
- Input: Any size (will be handled by model)
- Converts BGR (OpenCV) to RGB (PyTorch)
- No resizing needed (model handles it)

### Mask Processing
- Soft masks thresholded at 0.5
- Multiple masks combined using maximum operation
- Contours extracted for visualization

## 🔗 Related Scripts

- **Training**: `../training/train_maskrcnn_v1_2.py`
- **Data Generation**: `../generators/`
- **Validation**: `../validation/`

## 💡 Tips

1. **Always check overlay images** first to understand model behavior
2. **Adjust threshold** based on your use case (precision vs recall)
3. **Process similar images together** for consistent results
4. **Save checkpoints at different epochs** and compare inference results
5. **Use binary masks** for downstream tasks like text removal

## 📚 Requirements

```bash
pip install torch torchvision opencv-python numpy
```

GPU recommended for faster inference, but CPU works fine for small batches.
