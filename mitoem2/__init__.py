"""
MitoEM2: A comprehensive toolkit for mitochondria segmentation in EM images.

This package provides three main methods for mitochondria segmentation:
- MitoNet: Based on empanada framework
- MicroSAM: Based on Segment Anything Model
- nnUNet: Based on nnUNet framework
"""

__version__ = "0.1.0"

# Import main modules for easy access
from mitoem2.utils.logging import setup_logger
from mitoem2.utils.paths import get_project_root, get_data_root, get_checkpoint_dir

__all__ = [
    "__version__",
    "setup_logger",
    "get_project_root",
    "get_data_root",
    "get_checkpoint_dir",
]
