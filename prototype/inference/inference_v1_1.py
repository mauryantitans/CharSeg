"""
Inference Script v1.1
Character Instance Segmentation - Model Prediction & Visualization

Compatible with models trained using train_maskrcnn_v1_2.py
Auto-detects num_classes from checkpoint weights.

Features:
- Smart checkpoint loading (handles dict format)
- Automatic num_classes detection
- Binary mask output (all characters combined)
- Overlay visualization (bounding boxes + contours)
"""
import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# =====================================================
# 🔧 CONFIGURATION
# =====================================================
MODEL_PATH = "/content/drive/MyDrive/Project/PolyCHAR_rcnn_model-1.pth"  # trained model path
INPUT_DIR = "test_images"                  # folder with input images
OUTPUT_DIR = "test_results"                # folder to save outputs
SCORE_THRESHOLD = 0.5                      # confidence threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 🧠 MODEL LOADING
# =====================================================
print("🔍 Loading model checkpoint...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Detect if checkpoint is a dict with model_state
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint

# Try to infer number of classes from weights
num_classes = None
for k, v in state_dict.items():
    if "mask_fcn_logits.weight" in k:
        num_classes = v.shape[0]
        break

if num_classes is None:
    raise RuntimeError("❌ Could not detect number of classes from checkpoint.")
print(f"✅ Detected NUM_CLASSES = {num_classes}")

def get_model_instance_segmentation(num_classes):
    """
    Reconstruct Mask R-CNN model architecture.
    Must match training configuration.
    """
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 
        hidden_layer, 
        num_classes
    )
    
    return model

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {device}")

model = get_model_instance_segmentation(num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# =====================================================
# 🎨 IMAGE PREPROCESSING
# =====================================================
transform = T.Compose([T.ToTensor()])

# =====================================================
# 🔮 INFERENCE LOOP
# =====================================================
print(f"\n🚀 Starting inference on images in: {INPUT_DIR}")
print(f"📊 Score threshold: {SCORE_THRESHOLD}")

processed_count = 0

for img_name in os.listdir(INPUT_DIR):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"⚠️ Skipping {img_name} (failed to load)")
        continue

    # Convert to RGB for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model([img_tensor])

    # Extract predictions
    output = outputs[0]
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()

    # Initialize output images
    binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    overlay = image.copy()

    # Process each detection
    for i, score in enumerate(scores):
        if score < SCORE_THRESHOLD:
            continue
        
        # Convert soft mask to binary
        mask = (masks[i, 0] > 0.5).astype(np.uint8)
        
        # Combine with existing masks (union)
        binary_mask = np.maximum(binary_mask, mask * 255)

        # Draw bounding box on overlay
        color = (0, 255, 0)  # Green
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw mask contour
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)  # Yellow contours

    # Save outputs
    out_mask_path = os.path.join(OUTPUT_DIR, img_name)
    out_overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{img_name}")
    
    cv2.imwrite(out_mask_path, binary_mask)
    cv2.imwrite(out_overlay_path, overlay)

    processed_count += 1
    print(f"✅ Saved results for {img_name} ({len([s for s in scores if s >= SCORE_THRESHOLD])} detections)")

print(f"\n🎉 Inference complete!")
print(f"📁 Processed {processed_count} images")
print(f"💾 Results saved to: {OUTPUT_DIR}")
