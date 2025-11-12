# CharSeg Project Structure

## Directory Layout

```
CharSeg/
├── README.md                          # Project overview and setup instructions
├── CONCEPTS.md                        # Technical concepts documentation
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation script
│
├── configs/                          # Configuration files
│   ├── default.yaml                  # Default generation parameters
│   ├── simple_test.yaml              # Simple config for testing
│   └── full_generation.yaml          # Full dataset generation config
│
├── data_generation/                  # Core data generation modules
│   ├── __init__.py
│   ├── font_engine.py                # Character rendering and polygon extraction
│   ├── background_manager.py         # Background loading and analysis
│   ├── text_placer.py                # Text positioning and layout
│   ├── annotation_generator.py       # COCO format annotation creation
│   └── pipeline.py                   # End-to-end generation pipeline
│
├── utils/                            # Utility functions
│   ├── __init__.py
│   ├── visualization.py              # Visualization tools for debugging
│   ├── validation.py                 # Annotation validation functions
│   ├── topology.py                   # Topology checking (Euler number, etc.)
│   └── metrics.py                    # Statistics and quality metrics
│
├── datasets/                         # Data storage
│   ├── backgrounds/                  # Background images (10,000 images)
│   │   └── README.md                 # Instructions for downloading backgrounds
│   └── output/                       # Generated datasets
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── annotations/
│           ├── train.json
│           ├── val.json
│           └── test.json
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_test_font_engine.ipynb    # Test font rendering and contours
│   ├── 02_test_backgrounds.ipynb    # Explore background images
│   ├── 03_test_pipeline.ipynb       # Test end-to-end generation
│   └── 04_visualize_results.ipynb   # Visualize generated data
│
└── scripts/                          # Executable scripts
    ├── download_backgrounds.py       # Download background images
    ├── generate_dataset.py           # Main dataset generation script
    ├── validate_dataset.py           # Validate generated annotations
    └── visualize_samples.py          # Visualize random samples

```

## Implementation Order (Top-Down Approach)

### Phase 1: Minimal End-to-End (Week 1)
**Goal:** Generate 1 image with 1 character successfully

1. **utils/topology.py** - Basic topology functions
2. **data_generation/font_engine.py** - Minimal version
   - Render single character
   - Extract outer contour only (no holes yet)
3. **data_generation/background_manager.py** - Minimal version
   - Load one background image
4. **data_generation/text_placer.py** - Minimal version
   - Place single character at center
5. **data_generation/annotation_generator.py** - Minimal version
   - Create basic COCO JSON
6. **data_generation/pipeline.py** - Minimal version
   - Orchestrate: background + character → image + annotation
7. **configs/simple_test.yaml** - Minimal config
8. **scripts/generate_dataset.py** - Run pipeline
9. **utils/visualization.py** - Visualize result

**Milestone:** Successfully generate 1 image with annotation

### Phase 2: Add Topology Support (Week 1-2)
**Goal:** Properly handle characters with holes

1. **Enhance utils/topology.py**
   - Euler number calculation
   - Hole detection
   - Contour hierarchy processing
2. **Enhance font_engine.py**
   - Extract inner contours (holes)
   - Validate topology
3. **Enhance annotation_generator.py**
   - Add inner_segmentation field
   - Add topology metadata
4. **Test on topology-critical characters:** O, P, A, B, D

**Milestone:** Character 'O' has properly detected hole

### Phase 3: Scale to Words (Week 2)
**Goal:** Generate images with multiple characters forming words

1. **Enhance text_placer.py**
   - Multi-character placement
   - Proper character spacing (kerning-aware)
   - Word grouping
   - Collision detection (mask-based)
2. **Enhance annotation_generator.py**
   - word_id field
   - Multiple annotations per image
3. **Test with simple words:** "HELLO", "WORLD"

**Milestone:** Generate image with word "HELLO"

### Phase 4: Add Transformations (Week 2-3)
**Goal:** Add geometric transformations for data augmentation

1. **Enhance text_placer.py**
   - Rotation
   - Scaling
   - Perspective transformation
   - Transform polygons accordingly
2. **Enhance validation.py**
   - Verify topology preservation after transforms
3. **Test rotated text at various angles**

**Milestone:** Generate rotated text with correct annotations

### Phase 5: Batch Generation (Week 3)
**Goal:** Generate large datasets efficiently

1. **Enhance background_manager.py**
   - Load 10,000 backgrounds efficiently
   - Background caching
   - Background analysis (complexity, color distribution)
2. **Enhance pipeline.py**
   - Batch processing
   - Progress tracking
   - Error handling
   - Resume capability
3. **Create configs/full_generation.yaml**
4. **Optimize performance**
   - Multi-processing
   - Memory management

**Milestone:** Generate 1,000 images in reasonable time

### Phase 6: Quality & Validation (Week 3-4)
**Goal:** Ensure high-quality dataset

1. **Enhance utils/validation.py**
   - Comprehensive annotation checks
   - Statistical validation
2. **Enhance utils/metrics.py**
   - Dataset statistics
   - Character distribution
   - Quality metrics
3. **Create scripts/validate_dataset.py**
4. **Create visualization tools**

**Milestone:** Validated high-quality dataset

## File Descriptions

### Core Modules

#### `data_generation/font_engine.py`
```python
class FontEngine:
    """Renders characters and extracts precise polygon annotations"""
    
    def __init__(self, font_paths, size_range=(32, 128))
    def render_character(char, font, size, anti_alias=True) -> np.ndarray
    def extract_contours(mask) -> dict
        # Returns: {
        #   'outer_polygons': [...],
        #   'inner_polygons': [...],
        #   'bbox': [x, y, w, h],
        #   'has_hole': bool,
        #   'euler_number': int
        # }
    def get_random_font() -> str
```

#### `data_generation/background_manager.py`
```python
class BackgroundManager:
    """Manages background images for text placement"""
    
    def __init__(self, background_dir, cache_size=100)
    def load_backgrounds() -> list
    def get_random_background() -> np.ndarray
    def analyze_background(bg) -> dict
        # Returns: {
        #   'mean_color': [...],
        #   'complexity': float,
        #   'suitable_regions': [...]
        # }
```

#### `data_generation/text_placer.py`
```python
class TextPlacer:
    """Places text on backgrounds with collision detection"""
    
    def __init__(self, font_engine, collision_method='mask')
    def place_text(background, text, font, size, position=None) -> tuple
        # Returns: (composite_image, character_data)
    def calculate_spacing(chars, font, size) -> list
    def detect_collision(new_mask, existing_mask) -> bool
    def apply_transformation(image, polygons, transform_type) -> tuple
```

#### `data_generation/annotation_generator.py`
```python
class AnnotationGenerator:
    """Generates COCO-format annotations"""
    
    def __init__(self, output_path)
    def create_image_entry(image_id, filename, width, height) -> dict
    def create_annotation(annotation_id, image_id, char_data) -> dict
    def save_annotations(annotations, output_file)
```

#### `data_generation/pipeline.py`
```python
class DataGenerationPipeline:
    """End-to-end dataset generation orchestrator"""
    
    def __init__(self, config_path)
    def generate_single_image(image_id) -> tuple
    def generate_batch(start_id, count, split='train')
    def generate_dataset()
```

### Utility Modules

#### `utils/topology.py`
```python
def compute_euler_number(mask) -> int
def count_holes(mask) -> int
def extract_contour_hierarchy(contours, hierarchy) -> dict
def validate_topology(mask, expected_holes) -> bool
```

#### `utils/validation.py`
```python
def validate_polygon(polygon) -> bool
def validate_annotation(annotation) -> list  # Returns list of errors
def validate_dataset(annotation_file) -> dict  # Returns statistics
```

#### `utils/visualization.py`
```python
def visualize_annotation(image, annotation, show_topology=True)
def visualize_batch(images, annotations, grid_size=(4, 4))
def plot_polygon(ax, polygon, color, label)
def save_visualization(output_path)
```

#### `utils/metrics.py`
```python
def compute_dataset_statistics(annotation_file) -> dict
def character_distribution(annotations) -> dict
def check_topology_distribution(annotations) -> dict
def plot_statistics(stats, output_path)
```

## Configuration File Format

### `configs/default.yaml`
```yaml
# Dataset configuration
dataset:
  name: "CharSeg_v1"
  output_dir: "./datasets/output"
  
  splits:
    train: 50000
    val: 5000
    test: 5000

# Character configuration
characters:
  set: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  
# Font configuration
fonts:
  paths: ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/times.ttf"]
  size_range: [32, 128]
  anti_alias: true

# Text generation
text:
  words_per_image: [1, 5]  # Range
  chars_per_word: [3, 10]  # Range
  line_spacing: 1.5
  word_spacing: 2.0  # Multiplier of char spacing

# Background configuration
backgrounds:
  dir: "./datasets/backgrounds"
  image_size: [1024, 768]  # [width, height]
  
# Augmentation
augmentation:
  rotation: [-30, 30]  # Degrees
  scale: [0.8, 1.2]
  perspective: true
  
# Processing
processing:
  num_workers: 4
  batch_size: 100
  save_visualization: true
  visualization_samples: 10
```

## Development Workflow

### Setup
```bash
cd C:\Users\moury\OneDrive\Documents\GitHub\CharSeg
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Download Backgrounds
```bash
python scripts/download_backgrounds.py --count 10000
```

### Test Individual Components
```bash
# Test font engine
jupyter notebook notebooks/01_test_font_engine.ipynb

# Test backgrounds
jupyter notebook notebooks/02_test_backgrounds.ipynb
```

### Generate Dataset
```bash
# Test with simple config (1 image)
python scripts/generate_dataset.py --config configs/simple_test.yaml

# Generate full dataset
python scripts/generate_dataset.py --config configs/default.yaml
```

### Validate Dataset
```bash
python scripts/validate_dataset.py --annotations datasets/output/annotations/train.json
```

### Visualize Results
```bash
python scripts/visualize_samples.py --annotations datasets/output/annotations/train.json --num_samples 20
```

## Testing Strategy

### Unit Tests (per component)
1. **FontEngine**: Render 'A', check contours, verify holes
2. **BackgroundManager**: Load images, check dimensions
3. **TextPlacer**: Place single char, check collision detection
4. **AnnotationGenerator**: Create JSON, validate schema
5. **Topology**: Test Euler number on known characters

### Integration Tests
1. **Single character generation**: End-to-end test
2. **Multi-character generation**: Word placement
3. **Transformation test**: Rotated text with correct annotations
4. **Batch generation**: Generate 10 images

### Validation Tests
1. **Topology validation**: All 'O's have holes
2. **Annotation completeness**: All fields present
3. **Consistency**: Polygon contains bbox, bbox area matches polygon area

## Dependencies

### Core Dependencies
```
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
pyyaml>=6.0
```

### Visualization
```
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Utilities
```
tqdm>=4.65.0
```

### Optional (for advanced features)
```
shapely>=2.0.0  # For advanced polygon operations
scipy>=1.11.0   # For spatial operations
```

## Next Steps

1. **Review this structure** - Make sure it aligns with your vision
2. **Confirm decisions** - Any changes to the plan?
3. **Start coding** - Begin with Phase 1 (Minimal End-to-End)

Once you approve this structure, I'll start implementing the files in order!
