# Training Scripts

Model training code for character instance segmentation using Mask R-CNN.

## 📦 Version

### `train_maskrcnn_v1_2.py` - Training Script v1.2

Industry-standard Mask R-CNN training with multiple modes and checkpoint management.

## 🎯 Features

### Training Modes

1. **Train from Scratch**
   - Uses COCO pretrained backbone
   - Trains all layers from pretrained initialization
   ```python
   TRAIN_FROM_SCRATCH = True
   RESUME_TRAINING = False
   FINE_TUNE = False
   ```

2. **Resume Training**
   - Continues from saved checkpoint
   - Preserves optimizer state
   - Continues from last epoch number
   ```python
   TRAIN_FROM_SCRATCH = False
   RESUME_TRAINING = True
   FINE_TUNE = False
   CHECKPOINT_PATH = "path/to/checkpoint.pth"
   ```

3. **Fine-tuning** ⭐ RECOMMENDED for transfer learning
   - Loads pretrained weights
   - Reinitializes optimizer with lower LR (0.2x)
   - Best for adapting existing model to new data
   ```python
   TRAIN_FROM_SCRATCH = False
   RESUME_TRAINING = False
   FINE_TUNE = True
   CHECKPOINT_PATH = "path/to/checkpoint.pth"
   ```

### Checkpoint Management

- **Versioned Saving**: Creates `model_epoch_001.pth`, `model_epoch_002.pth`, etc.
- **Every Epoch**: Optional saving after each epoch
- **Final Save**: Always saves final model
- **Stored Info**: epoch, model_state, optimizer_state, seed

### Reproducibility

- Fixed random seeds (torch, numpy, CUDA)
- Seed stored in checkpoints
- Deterministic training when possible

## 🚀 Quick Start

### 1. Configure Paths

Edit the configuration section:

```python
# Data directory (should contain images/ and annotations/ folders)
DATA_DIR = "/path/to/your/dataset"

# Where to save trained model
MODEL_SAVE_PATH = "/path/to/save/model.pth"

# Optional: pretrained checkpoint for resume/fine-tune
CHECKPOINT_PATH = "/path/to/pretrained.pth"
```

### 2. Set Training Mode

Choose ONE of the three modes:

```python
# For first training run
TRAIN_FROM_SCRATCH = True
RESUME_TRAINING = False
FINE_TUNE = False

# OR for continuing interrupted training
TRAIN_FROM_SCRATCH = False
RESUME_TRAINING = True
FINE_TUNE = False

# OR for adapting to new dataset
TRAIN_FROM_SCRATCH = False
RESUME_TRAINING = False
FINE_TUNE = True
```

### 3. Adjust Hyperparameters

```python
NUM_EPOCHS = 10              # Number of epochs to train
BATCH_SIZE = 2               # Images per batch (adjust based on GPU)
LEARNING_RATE = 0.005        # Initial learning rate
SEED = 15107                 # Random seed for reproducibility
```

### 4. Run Training

```bash
python train_maskrcnn_v1_2.py
```

## 📊 Dataset Format

Expected directory structure:

```
DATA_DIR/
├── images/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── annotations/
    ├── 00001.csv
    ├── 00002.csv
    └── ...
```

CSV format:
```csv
char,polygon_json
A,"[x1,y1,x2,y2,x3,y3,...]"
B,"[x1,y1,x2,y2,x3,y3,...]"
```

## 🎛️ Advanced Configuration

### Saving Options

```python
SAVE_EVERY_EPOCH = True      # Save after each epoch
SAVE_VERSIONED = True        # Create versioned files
```

### Learning Rate Schedule

The script uses StepLR scheduler:
- Reduces LR by 0.1x every 3 epochs
- Can be modified in the code

### Optimizer

Default: SGD with momentum
- Learning rate: 0.005
- Momentum: 0.9
- Weight decay: 0.0005

For fine-tuning, LR is automatically reduced to 0.2x initial value.

## 📈 Monitoring

Training progress shows:
- Current epoch
- Batch-level loss
- Epoch average loss
- Checkpoint save confirmations

Example output:
```
Using device: cuda
🚀 Starting training...
Epoch 1/10: 100%|████████| 500/500 [12:34<00:00, loss=0.8234]
✅ Epoch 1 finished. Avg Loss: 0.8234
💾 Saved checkpoint: model_epoch_001.pth
```

## 🔧 Troubleshooting

### Out of Memory Error
- Reduce `BATCH_SIZE`
- Use smaller images
- Use fewer workers (`num_workers=0`)

### No valid polygons warning
- Check annotation CSV format
- Verify polygon coordinates are valid
- Some images may have no annotations (will be skipped)

### Checkpoint loading fails
- Verify checkpoint path exists
- Check num_classes matches (2 for binary: background + character)
- Ensure checkpoint format is compatible (v1.2 format)

## 📝 Model Architecture

- **Base**: Mask R-CNN with ResNet-50 FPN backbone
- **Input**: RGB images (any size, will be resized internally)
- **Output**: 
  - Bounding boxes
  - Class labels (background=0, character=1)
  - Instance segmentation masks

## 🎯 Performance Tips

1. **Start with fine-tuning**: If you have a pretrained model, fine-tuning is much faster
2. **Use GPU**: Training on CPU is extremely slow
3. **Monitor loss**: Should decrease steadily; if not, adjust learning rate
4. **Save frequently**: Enable `SAVE_EVERY_EPOCH` to avoid losing progress
5. **Use versioned saves**: Helps compare different checkpoints

## 📚 Requirements

```bash
pip install torch torchvision opencv-python numpy pillow tqdm
```

GPU recommended for reasonable training speed.

## 🔗 Related Scripts

- **Font Download**: `../font_download/`
- **Data Generation**: `../generators/`
- **Validation**: `../validation/`
- **Inference**: `../inference/`
