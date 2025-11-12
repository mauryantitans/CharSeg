# -*- coding: utf-8 -*-
"""
Mask R-CNN Training Script v1.2
Character Instance Segmentation Training with Resume/Fine-tune Support

Features:
- Train from scratch with COCO pretrained backbone
- Resume training from checkpoint (continues same run)
- Fine-tune pretrained model on new dataset
- Versioned checkpoint saving
- Reproducible training with fixed seeds
"""
import os
import json
import csv
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm

# =====================================================
# 🔧 CONFIGURATION
# =====================================================
DATA_DIR = "/content/drive/MyDrive/Project/Dataset-Synthetic GenV-2"

# Base model save path
MODEL_SAVE_PATH = "/content/drive/MyDrive/Project/PolyCHAR_rcnn_model_v2-1.pth"

# Optional: pretrained or previous model path
CHECKPOINT_PATH = "/content/drive/MyDrive/Project/PolyCHAR_rcnn_model-2(1).pth"

# Modes
TRAIN_FROM_SCRATCH = False
RESUME_TRAINING = False      # resume same run with optimizer
FINE_TUNE = True             # fine-tune pretrained model on new dataset

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.005
SEED = 15107

# Saving options
SAVE_EVERY_EPOCH = True
SAVE_VERSIONED = True

# =====================================================
# 🧩 REPRODUCIBILITY
# =====================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# =====================================================
# 📦 DATASET DEFINITION
# =====================================================
class CharDataset(Dataset):
    """
    Character Instance Segmentation Dataset
    
    Loads images and polygon annotations from CSV files.
    Converts polygons to binary masks and bounding boxes.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.imgs_dir = os.path.join(root_dir, "images")
        self.annos_dir = os.path.join(root_dir, "annotations")
        self.image_files = sorted([f for f in os.listdir(self.imgs_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.imgs_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        anno_path = os.path.join(self.annos_dir, img_name.replace('.png', '.csv'))
        boxes, masks, labels = [], [], []
        img_h, img_w = image.shape[1], image.shape[2]

        try:
            with open(anno_path, newline='', encoding='utf8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        poly = json.loads(row["polygon_json"])
                        pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
                        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
                        if x_max <= x_min or y_max <= y_min:
                            continue
                        mask = np.zeros((img_h, img_w), dtype=np.uint8)
                        cv2.fillPoly(mask, [pts], 1)
                        masks.append(mask)
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[WARN] Missing annotation: {anno_path} ({e})")
            return None

        if not masks:
            print(f"[WARN] Skipping {img_name} (no valid polygons)")
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        return image, target

def collate_fn(batch):
    """Custom collate function to handle None samples"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))

# =====================================================
# 🧠 MODEL SETUP
# =====================================================
def get_model_instance_segmentation(num_classes):
    """
    Create Mask R-CNN model with custom predictors.
    
    Args:
        num_classes: Number of classes (including background)
    
    Returns:
        Mask R-CNN model with ResNet-50 FPN backbone
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# =====================================================
# 🚀 MAIN TRAINING LOOP
# =====================================================
if __name__ == "__main__":
    num_classes = 2  # background + character
    
    # Load dataset
    dataset = CharDataset(root_dir=DATA_DIR)
    
    # Train/test split
    indices = torch.randperm(len(dataset)).tolist()
    test_size = min(50, int(0.1 * len(dataset)))
    dataset_train = torch.utils.data.Subset(dataset, indices[:-test_size])
    dataset_test = torch.utils.data.Subset(dataset, indices[-test_size:])

    # Data loaders
    data_loader = DataLoader(
        dataset_train, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collate_fn
    )

    # Initialize model
    model = get_model_instance_segmentation(num_classes)
    model.to(DEVICE)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    start_epoch = 0

    # =================================================
    # ♻️ LOAD / RESUME / FINE-TUNE
    # =================================================
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"🔁 Resumed from checkpoint: {CHECKPOINT_PATH} (epoch {start_epoch})")

    elif FINE_TUNE and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        # Reinitialize optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE * 0.2, 
            momentum=0.9, 
            weight_decay=0.0005
        )
        print(f"✅ Loaded pretrained weights from {CHECKPOINT_PATH} for fine-tuning.")

    elif TRAIN_FROM_SCRATCH:
        print("🆕 Training from scratch (no checkpoint loaded).")
    else:
        print("⚠️ No checkpoint found — training from scratch by default.")

    # =================================================
    # 🏋️ TRAIN LOOP
    # =================================================
    print("🚀 Starting training...")
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{start_epoch+NUM_EPOCHS}")

        for batch in progress_bar:
            if batch[0] is None:
                continue
            
            images, targets = batch
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_loss = running_loss / len(data_loader)
        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint after every epoch
        if SAVE_EVERY_EPOCH:
            if SAVE_VERSIONED:
                epoch_save_path = f"{os.path.splitext(MODEL_SAVE_PATH)[0]}_epoch_{epoch+1:03d}.pth"
            else:
                epoch_save_path = MODEL_SAVE_PATH
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'seed': SEED
            }, epoch_save_path)
            print(f"💾 Saved checkpoint: {epoch_save_path}")

    # =================================================
    # 🏁 FINAL SAVE
    # =================================================
    torch.save({
        'epoch': start_epoch + NUM_EPOCHS - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'seed': SEED
    }, MODEL_SAVE_PATH)
    print(f"🎯 Training complete. Final model saved to {MODEL_SAVE_PATH}")
