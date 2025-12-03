# CharSeg: Topology-Aware Character Instance Segmentation

## Project Overview

A machine learning system for precise character-level instance segmentation that preserves topological features (holes in characters like O, P, Q, R, A, B, D) and provides hierarchical understanding from characters to words. The primary goal is to enable pixel-perfect text removal from images without disturbing surrounding content.



## Project Status

**Current Phase:** Phase 3 - Adding Real world blur noise to data and training using the noisy data

### Completed
- [x] Prototype Buit and tested 
- [x] Prototype Documentation (README, CONCEPTS, PROJECT_STRUCTURE)
- [x] Minimal end-to-end pipeline
- [x] Font engine implementation
- [x] Topology support (holes detection)
- [x] Multi-character placement
- [x] Geometric transformations
- [x] Model training

### In Progress
- [ ] Background manager

### Planned
- [ ] Batch generation
- [ ] Transforming Prototype into Viable Real-World Solution
- [ ] Research paper

## Documentation

- **[README.md](README.md)** - This file, project overview and setup
- **[CONCEPTS.md](CONCEPTS.md)** - Deep dive into technical concepts
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed implementation plan

## Key Features 

- ✅ **Topology-Aware**: Preserves holes in characters (O, P, A, etc.)
- ✅ **Hierarchical**: Character-level + word-level understanding
- ✅ **Pixel-Perfect**: Polygon annotations, not just bounding boxes
- ✅ **Realistic**: 10,000 complex backgrounds for training
- ✅ **Flexible**: Handles rotation, perspective, various fonts
- ✅ **Research-Quality**: Publication-ready code and experiments

## Research Objectives

1. Develop topology-preserving character segmentation
2. Enable hierarchical text understanding (character + word level)
3. Create a synthetic dataset generator with precise polygon annotations
4. Build a model capable of pixel-perfect text detection and removal
5. Publish research findings on character-level segmentation

## Contact

**Author:** Mourya
**Status:** Active Development (November 2025)
**Goal:** Research publication on character instance segmentation

---

## Quick Start (Check Prototype Folder)

```bash
# 1. Setup environment
cd C:\Users\moury\OneDrive\Documents\GitHub\CharSeg
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Download background images (10,000 images)
python scripts/download_backgrounds.py --count 10000

# 3. Generate test dataset
python scripts/generate_dataset.py --config configs/simple_test.yaml

# 4. Validate generated data
python scripts/validate_dataset.py --annotations datasets/output/annotations/train.json

# 5. Visualize samples
python scripts/visualize_samples.py --annotations datasets/output/annotations/train.json --num_samples 20
```

For detailed implementation plan, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
For technical concepts, see [CONCEPTS.md](CONCEPTS.md)
