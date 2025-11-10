"""
Example script demonstrating how to use fuse_adjacent_blocks function.

This example shows how to merge two adjacent blocks that were predicted separately.
For instance, if you have a volume of (512, 1024, 1024) and split it into:
  - Block 1: shape (256, 1024, 1024), representing z=0:256
  - Block 2: shape (256, 1024, 1024), representing z=256:512

The function will merge them by computing IoU between the interface slices.
"""

import numpy as np
import tifffile
from fuse_segs_overlap import fuse_adjacent_blocks

def create_example_blocks():
    """
    Create two example blocks with some overlapping labels at the interface.
    This is just for demonstration purposes.
    """
    # Block 1: (256, 1024, 1024)
    block1 = np.zeros((256, 1024, 1024), dtype=np.int16)
    # Add some example labels
    block1[100:150, 400:600, 400:600] = 1  # Label 1
    block1[200:256, 300:500, 300:500] = 2  # Label 2 (extends to the boundary)
    
    # Block 2: (256, 1024, 1024)
    block2 = np.zeros((256, 1024, 1024), dtype=np.int16)
    # Label that should match with block1's label 2
    block2[0:100, 300:500, 300:500] = 1  # This should merge with block1's label 2
    # A new label that doesn't overlap
    block2[150:200, 700:800, 700:800] = 2
    
    return block1, block2

def example_usage_with_files():
    """
    Example: Load from files and fuse
    """
    # Assuming you have two prediction files
    block1_file = "path/to/block1_prediction.tif"
    block2_file = "path/to/block2_prediction.tif"
    output_file = "path/to/fused_output.tif"
    
    # Fuse the blocks
    fused = fuse_adjacent_blocks(
        block1=block1_file,
        block2=block2_file,
        save_file=output_file,
        filter_size=100,  # Remove components smaller than 100 voxels
        overlap_thresh=0.3,  # IoU threshold for merging labels
        is_reset_label_ids=True  # Reset label IDs to be sequential
    )
    
    print(f"Fused segmentation saved to {output_file}")
    print(f"Final shape: {fused.shape}")
    print(f"Number of unique labels: {len(np.unique(fused)) - 1}")  # -1 for background

def example_usage_with_arrays():
    """
    Example: Use numpy arrays directly
    """
    # Create example blocks
    block1, block2 = create_example_blocks()
    
    print(f"Block 1 shape: {block1.shape}, unique labels: {np.unique(block1)}")
    print(f"Block 2 shape: {block2.shape}, unique labels: {np.unique(block2)}")
    
    # Fuse the blocks
    fused = fuse_adjacent_blocks(
        block1=block1,
        block2=block2,
        save_file="example_fused.tif",
        filter_size=50,
        overlap_thresh=0.3,
        is_reset_label_ids=True
    )
    
    print(f"Fused shape: {fused.shape}")
    print(f"Unique labels in fused: {np.unique(fused)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Example: Fusing adjacent blocks with numpy arrays")
    print("=" * 60)
    example_usage_with_arrays()
    
    print("\n" + "=" * 60)
    print("For file-based usage, see example_usage_with_files()")
    print("=" * 60)

