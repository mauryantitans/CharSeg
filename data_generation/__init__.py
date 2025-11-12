"""
CharSeg Data Generation Module

This module contains all components for synthetic text dataset generation:
- FontEngine: Character rendering and polygon extraction
- BackgroundManager: Background image loading and analysis
- TextPlacer: Text positioning and collision detection
- AnnotationGenerator: COCO format annotation creation
- Pipeline: End-to-end generation orchestration
"""

__version__ = "0.1.0"
__author__ = "Mourya"

from .font_engine import FontEngine
from .background_manager import BackgroundManager
from .text_placer import TextPlacer
from .annotation_generator import AnnotationGenerator
from .pipeline import DataGenerationPipeline

__all__ = [
    "FontEngine",
    "BackgroundManager",
    "TextPlacer",
    "AnnotationGenerator",
    "DataGenerationPipeline"
]
