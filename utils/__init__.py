"""
CharSeg Utilities Module

Contains helper functions for:
- Topology analysis (Euler numbers, hole detection)
- Validation (annotation checking)
- Visualization (debugging tools)
- Metrics (dataset statistics)
"""

__version__ = "0.1.0"

from .topology import compute_euler_number, count_holes, validate_topology
from .validation import validate_polygon, validate_annotation, validate_dataset
from .visualization import visualize_annotation, visualize_batch
from .metrics import compute_dataset_statistics

__all__ = [
    "compute_euler_number",
    "count_holes",
    "validate_topology",
    "validate_polygon",
    "validate_annotation",
    "validate_dataset",
    "visualize_annotation",
    "visualize_batch",
    "compute_dataset_statistics"
]
