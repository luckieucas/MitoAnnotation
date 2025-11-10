#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of mask_3d_visualizer.py
This script demonstrates how to use the simplified bounding box visualizer.
Each label gets one page in a single PDF.
"""

import os
import sys
import subprocess

def run_visualizer_example():
    """Run the visualizer with example parameters."""
    
    # Example command
    cmd = [
        "python", "mask_3d_visualizer.py",
        "--input", "your_mask_file.tiff",  # Replace with actual file
        "--output", "output_visualization.pdf",
        "--views", "8",
        "--size", "800", "600",
        "--smooth", "30",
        "--spacing", "1.0", "1.0", "1.0",
        "--title", "3D Mask Visualization"
    ]
    
    print("Example usage:")
    print(" ".join(cmd))
    print()
    
    print("Available options:")
    print("  --input, -i     : Input 3D mask file (TIFF format)")
    print("  --output, -o    : Output PDF file path")
    print("  --spacing       : Voxel spacing z,y,x (default: 1.0 1.0 1.0)")
    print("  --views, -v     : Number of viewpoints (default: 8)")
    print("  --size          : Image size width height (default: 800 600)")
    print("  --title, -t     : PDF title (default: '3D Visualization')")
    print("  --max-labels    : Maximum number of labels to process (for testing)")
    print()
    
    print("Usage Examples:")
    print("# Basic usage: Each label gets one page in a single PDF")
    print("python mask_3d_visualizer.py -i mask.tiff -o result.pdf")
    print()
    print("# High resolution with 12 views:")
    print("python mask_3d_visualizer.py -i mask.tiff -o result.pdf --views 12 --size 1200 900")
    print()
    print("# Test with limited labels:")
    print("python mask_3d_visualizer.py -i mask.tiff -o result.pdf --max-labels 5")
    print()
    print("# Custom spacing:")
    print("python mask_3d_visualizer.py -i mask.tiff -o result.pdf --spacing 0.5 0.5 1.0")
    print()
    print("# Custom title:")
    print("python mask_3d_visualizer.py -i mask.tiff -o result.pdf --title 'My 3D Analysis'")

if __name__ == "__main__":
    run_visualizer_example()
