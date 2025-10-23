"""
Segmentation Fusion Utilities

This module provides functions to fuse segmentation masks from different sources:
1. Fuse overlapping segmentations from different directions
2. Fuse multiple segmentations sequentially
3. Fuse adjacent blocks along z-axis

Example usage for fusing adjacent blocks:
    python fuse_segs_overlap.py --fuse_adjacent_blocks \
        --pred_file1 block1_pred.tif \
        --pred_file2 block2_pred.tif \
        --save_name fused_result.tif \
        --overlap_thresh 0.3 \
        --filter_size 100 \
        --reset_label_ids

For a volume of size (512, 1024, 1024) split into two blocks of (256, 1024, 1024):
    - Block 1: z=0:256
    - Block 2: z=256:512
    The function computes IoU between the last slice of Block 1 and first slice of Block 2
    to merge matching labels and ensure unique labels in the final volume.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import tifffile
import numpy as np
from tqdm import tqdm
from typing import Union
from em_util.seg import seg_to_iou
import scipy.ndimage as ndimage

MAX_LABEL = 2000
OFFSET_LABEL = 1000

def remove_small_predictions(pred_mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Remove small connected components from a 3D prediction mask.

    Parameters:
        pred_mask (np.ndarray): 3D prediction mask.
        min_size (int): Minimum volume (number of voxels) required to keep a component.

    Returns:
        np.ndarray: Filtered mask with small components removed.
    """
    unique_labels = np.unique(pred_mask)
    for label in unique_labels:
        if label == 0:
            continue
        if (pred_mask == label).astype(np.uint8).sum() < min_size:
            pred_mask[pred_mask == label] = 0
    return pred_mask

def reset_mask_label_id(mask_image: np.ndarray) -> np.ndarray:
    """
    Reset the label IDs in the segmentation mask to ensure sequential order.

    Parameters:
        mask_image (np.ndarray): Segmentation mask with potentially unordered labels.

    Returns:
        np.ndarray: Segmentation mask with reset (sequential) label IDs.
    """
    print("Resetting mask label IDs")
    label_list = list(range(1, MAX_LABEL))
    for label in sorted(np.unique(mask_image)):
        if label == 0:
            continue
        mask_image[mask_image == label] = label_list.pop(0)
    return mask_image

def process_intersection_mask(x_seg: np.ndarray, y_seg: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Process the intersection between two segmentation masks. For overlapping labels, add
    components from y_seg into x_seg (where x_seg is background) and remove them from y_seg.

    Parameters:
        x_seg (np.ndarray): Segmentation from one direction.
        y_seg (np.ndarray): Segmentation from another direction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The updated x_seg and y_seg.
    """
    x_labels = np.unique(x_seg)
    y_labels = np.unique(y_seg)
    intersection_labels = np.intersect1d(x_labels, y_labels)
    print(f"Intersection labels: {intersection_labels}")
    for label in intersection_labels:
        if label == 0:
            continue
        x_seg[(x_seg == 0) & (y_seg == label)] = label
        y_seg[y_seg == label] = 0
    return x_seg, y_seg

def load_segmentation(seg_input: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load segmentation from a file path or return it if it is already a numpy array.

    Parameters:
        seg_input (str or np.ndarray): File path to a tif image or a numpy array.

    Returns:
        np.ndarray: Loaded segmentation mask.
    """
    if isinstance(seg_input, str):
        return tifffile.imread(seg_input).astype(np.int16)
    return seg_input

def fuse_segmentations(x_seg: np.ndarray, y_seg: np.ndarray,
                           filter_size: int = 500, overlap_thresh: float = 0.2,
                           is_reset_label_ids: bool = True) -> np.ndarray:
    """
    Fuse two segmentation masks into one.

    Parameters:
        x_seg (np.ndarray): First segmentation mask.
        y_seg (np.ndarray): Second segmentation mask.
        filter_size (int): Minimum volume required to keep a component.
        overlap_thresh (float): Overlap ratio threshold for merging components.
        is_reset_label_ids (bool): Whether to reset label IDs after fusion.

    Returns:
        np.ndarray: The fused segmentation mask.
    """
    # Process intersection or apply offset to avoid label conflicts
    if isinstance(x_seg, str):
        x_seg = load_segmentation(x_seg)
    if isinstance(y_seg, str):
        y_seg = load_segmentation(y_seg)
    if not is_reset_label_ids:
        x_seg, y_seg = process_intersection_mask(x_seg, y_seg)
    else:
        y_seg = y_seg + OFFSET_LABEL
        y_seg[y_seg == OFFSET_LABEL] = 0

    # Remove small connected components
    x_seg = remove_small_predictions(x_seg, min_size=filter_size)
    y_seg = remove_small_predictions(y_seg, min_size=filter_size)

    # Compute IoU between segmentation masks
    outx_y = seg_to_iou(x_seg, y_seg)
    outy_x = seg_to_iou(y_seg, x_seg)

    # Merge labels based on IoU criteria
    for pair in outy_x:
        # If there is no corresponding label in x_seg and the component is small, remove it
        if pair[1] == 0 and pair[2] < filter_size:
            y_seg[y_seg == pair[0]] = 0
        # If sufficient overlap, merge y_seg label into x_seg label
        elif pair[1] != 0 and pair[4] / (pair[2] + pair[3] - pair[4]) > overlap_thresh:
            print(f"Matched y_x: {pair}")
            y_seg[y_seg == pair[0]] = pair[1]

    for pair in outx_y:
        if pair[1] != 0 and pair[4] / (pair[2] + pair[3] - pair[4]) > overlap_thresh:
            print(f"Matched x_y: {pair}")

    # Fuse the two segmentations: where x_seg is background, take value from y_seg
    fused_seg = np.where(x_seg == 0, y_seg, x_seg)

    # Reset label IDs if required
    if is_reset_label_ids:
        fused_seg = reset_mask_label_id(fused_seg)

    return fused_seg

def fuse_segmentation_list(pred_list: list, save_file: str = "res_tif/pred_mask_all.tif",
                             filter_size: int = 500, overlap_thresh: float = 0.2,
                             is_reset_label_ids: bool = True) -> np.ndarray:
    """
    Fuse a list of segmentation predictions sequentially.

    Parameters:
        pred_list (list): List of file paths or numpy arrays for segmentation predictions.
        save_file (str): Path to save the final fused segmentation mask.
        filter_size (int): Minimum volume required to keep a component.
        overlap_thresh (float): Overlap ratio threshold for merging components.
        is_reset_label_ids (bool): Whether to reset label IDs after fusion.

    Returns:
        np.ndarray: The final fused segmentation mask.
    """
    if not pred_list:
        raise ValueError("The prediction file list is empty.")

    # Load the first segmentation as the initial fused result
    fused_seg = load_segmentation(pred_list[0])

    # Sequentially fuse each additional segmentation into the current result
    for pred in tqdm(pred_list[1:], desc="Fusing segmentations"):
        next_seg = load_segmentation(pred)
        fused_seg = fuse_segmentations(fused_seg, next_seg,
                                           filter_size=filter_size,
                                           overlap_thresh=overlap_thresh,
                                           is_reset_label_ids=is_reset_label_ids)

    # Save the final fused segmentation
    tifffile.imwrite(save_file, fused_seg.astype(np.int16))
    return fused_seg

def fuse_adjacent_blocks(block1: Union[str, np.ndarray],
                         block2: Union[str, np.ndarray],
                         save_file: str = None,
                         filter_size: int = 100,
                         overlap_thresh: float = 0.2,
                         is_reset_label_ids: bool = True) -> np.ndarray:
    """
    Fuse two adjacent blocks along the z-axis by matching labels at their interface.
    
    This function merges two 3D segmentation blocks that are adjacent along the z-axis.
    It computes IoU between the last slice of block1 and the first slice of block2
    to determine which labels should be merged.
    
    Parameters:
        block1 (str or np.ndarray): First block segmentation (top block).
        block2 (str or np.ndarray): Second block segmentation (bottom block).
        save_file (str): Path to save the fused segmentation mask (optional).
        filter_size (int): Minimum volume required to keep a component.
        overlap_thresh (float): IoU threshold for merging labels.
        is_reset_label_ids (bool): Whether to reset label IDs after fusion.
    
    Returns:
        np.ndarray: The fused segmentation mask combining both blocks.
    """
    # Load segmentations
    block1_seg = load_segmentation(block1)
    block2_seg = load_segmentation(block2)
    
    print(f"Block1 shape: {block1_seg.shape}")
    print(f"Block2 shape: {block2_seg.shape}")
    
    # Remove small components from both blocks
    block1_seg = remove_small_predictions(block1_seg, min_size=filter_size)
    block2_seg = remove_small_predictions(block2_seg, min_size=filter_size)
    
    # Add offset to block2 labels to avoid conflicts
    max_label_block1 = block1_seg.max()
    print(f"Max label in block1: {max_label_block1}")
    
    block2_seg_offset = block2_seg.copy()
    block2_seg_offset[block2_seg_offset > 0] += max_label_block1
    
    # Extract interface slices
    last_slice_block1 = block1_seg[-1, :, :]
    first_slice_block2 = block2_seg[0, :, :]
    
    print(f"Computing IoU between interface slices...")
    # Compute IoU between the two interface slices
    # seg_to_iou returns: [[label1, label2, size1, size2, intersection_size], ...]
    iou_block2_to_block1 = seg_to_iou(first_slice_block2, last_slice_block1)
    
    # Create label mapping for block2
    label_mapping = {}  # old_label -> new_label
    
    for pair in iou_block2_to_block1:
        block2_label = pair[0]
        block1_label = pair[1]
        size_block2 = pair[2]
        size_block1 = pair[3]
        intersection = pair[4]
        
        if block2_label == 0:
            continue
            
        # Calculate IoU
        if size_block2 + size_block1 - intersection > 0:
            iou = intersection / (size_block2 + size_block1 - intersection)
        else:
            iou = 0
        
        # If IoU exceeds threshold and block1 has a matching label, merge them
        if block1_label != 0 and iou > overlap_thresh:
            label_mapping[block2_label] = block1_label
            print(f"Merging: block2 label {block2_label} -> block1 label {block1_label} (IoU: {iou:.3f})")
    
    # Apply label mapping to block2
    block2_seg_merged = block2_seg_offset.copy()
    for old_label, new_label in label_mapping.items():
        # Map from original block2 label to block1 label
        block2_seg_merged[block2_seg == old_label] = new_label
    
    # Concatenate the two blocks along z-axis
    fused_seg = np.concatenate([block1_seg, block2_seg_merged], axis=0)
    
    print(f"Fused segmentation shape: {fused_seg.shape}")
    print(f"Number of unique labels before reset: {len(np.unique(fused_seg)) - 1}")  # -1 for background
    
    # Reset label IDs if required
    if is_reset_label_ids:
        fused_seg = reset_mask_label_id(fused_seg)
        print(f"Number of unique labels after reset: {len(np.unique(fused_seg)) - 1}")
    
    # Save if path is provided
    if save_file:
        tifffile.imwrite(save_file, fused_seg.astype(np.int16))
        print(f"Saved fused segmentation to: {save_file}")
    
    return fused_seg

def fuse_segmentations_different_directions(x_seg_tif: Union[str, np.ndarray],
                                             y_seg_tif: Union[str, np.ndarray],
                                             z_seg_tif: Union[str, np.ndarray],
                                             save_file: str = "res_tif/pred_mask_all.tif",
                                             filter_size: int = 500,
                                             overlap_thresh: float = 0.2,
                                             is_reset_label_ids: bool = True) -> np.ndarray:
    """
    Fuse segmentation results from three directions into a single segmentation mask.

    Parameters:
        x_seg_tif, y_seg_tif, z_seg_tif (str or np.ndarray): Segmentation masks from different directions.
        save_file (str): Path to save the fused segmentation mask.
        filter_size (int): Minimum volume required to keep a component.
        overlap_thresh (float): Overlap ratio threshold for merging components.
        is_reset_label_ids (bool): Whether to reset label IDs after fusion.

    Returns:
        np.ndarray: The fused segmentation mask.
    """
    print(f"is_reset_label_ids: {is_reset_label_ids}")

    # Load segmentation masks
    x_seg = load_segmentation(x_seg_tif)
    y_seg = load_segmentation(y_seg_tif)
    z_seg = load_segmentation(z_seg_tif)

    # The function process_intersection_mask_xyz is assumed to be defined elsewhere.
    if not is_reset_label_ids:
        x_seg, y_seg, z_seg = process_intersection_mask_xyz(x_seg, y_seg, z_seg)
    else:
        y_seg = y_seg + OFFSET_LABEL
        y_seg[y_seg == OFFSET_LABEL] = 0

    x_seg = remove_small_predictions(x_seg, min_size=filter_size)
    y_seg = remove_small_predictions(y_seg, min_size=filter_size)

    outx_y = seg_to_iou(x_seg, y_seg)
    outy_x = seg_to_iou(y_seg, x_seg)
    print("outx_y:", outx_y)
    print("*" * 50)
    print("outy_x:", outy_x)
    y_x_matched = {}
    for pair in outy_x:
        if pair[1] == 0 and pair[2] < filter_size:
            y_seg[y_seg == pair[0]] = 0
        elif pair[1] != 0 and pair[4] / (pair[2] + pair[3] - pair[4]) > overlap_thresh:
            print(f"Matched y_x: {pair}")
            y_x_matched.setdefault(pair[1], []).append([pair[0], pair[4] / (pair[2] + pair[3] - pair[4])])
    for key, value in y_x_matched.items():
        if len(value) > 1:
            print(f"Merging label: {key}, values: {value}")
            total_iou = sum(v[1] for v in value)
            if total_iou > 0.7:
                x_seg[x_seg == key] = 0
        else:
            print(f"Key: {key}, value: {value}")
            y_seg[y_seg == value[0][0]] = key

    fused_seg = np.where(x_seg == 0, y_seg, x_seg)
    if is_reset_label_ids:
        fused_seg = reset_mask_label_id(fused_seg)

    print(f"Unique fused_seg labels: {np.unique(fused_seg)}")
    tifffile.imwrite(save_file, fused_seg.astype(np.int16))
    return fused_seg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=str, default="hela_cell_em")
    parser.add_argument("--anno_interval", type=int, default=102)
    # New argument: list of prediction file names (can be of any length)
    parser.add_argument("--pred_files", nargs="+", help="List of prediction file names", default=[])
    parser.add_argument("--pred_file1", type=str, default="res_tif/pred_mask_all.tif")
    parser.add_argument("--pred_file2", type=str, default="res_tif/pred_mask_all.tif")
    parser.add_argument("--pred_file3", type=str, default="res_tif/pred_mask_all.tif")
    parser.add_argument("--save_name", type=str, default="res_tif")
    parser.add_argument("--overlap_thresh", type=float, default=0.5)
    parser.add_argument("--filter_size", type=int, default=100, help="Minimum volume to keep a component")
    parser.add_argument("--reset_label_ids", action="store_true", default=False)
    parser.add_argument("--fuse_different_directions", action="store_true", default=False)
    parser.add_argument("--fuse_adjacent_blocks", action="store_true", default=False,
                       help="Fuse two adjacent blocks along z-axis")
    args = parser.parse_args()

    cell = args.cell
    anno_interval = args.anno_interval

    res_path = os.path.join('../results/', cell, f'interval{anno_interval}')
    gt_file = f'/mmfs1/data/liupen/project/dataset/mito/3dem_labeled/label/{cell.replace("_em", "_mito")}.tif'
    save_path = os.path.join(res_path, args.save_name)
    print(f"Arguments: {args}")

    # If a list of prediction files is provided, fuse them sequentially.
    if args.pred_files:
        pred_list = [os.path.join(res_path, pf) for pf in args.pred_files]
        fuse_segmentation_list(pred_list, save_file=save_path,
                               overlap_thresh=args.overlap_thresh,
                               is_reset_label_ids=args.reset_label_ids)
    elif args.fuse_adjacent_blocks:
        # Fuse two adjacent blocks along z-axis
        pred_file1 = os.path.join(res_path, args.pred_file1)
        pred_file2 = os.path.join(res_path, args.pred_file2)
        print(f"Fusing adjacent blocks:")
        print(f"  Block 1: {pred_file1}")
        print(f"  Block 2: {pred_file2}")
        fused_seg = fuse_adjacent_blocks(pred_file1, pred_file2,
                                         save_file=save_path,
                                         filter_size=args.filter_size,
                                         overlap_thresh=args.overlap_thresh,
                                         is_reset_label_ids=args.reset_label_ids)
    else:
        if not args.fuse_different_directions:
            pred_file1 = os.path.join(res_path, args.pred_file1)
            pred_file2 = os.path.join(res_path, args.pred_file2)
            # For backwards compatibility, use the two-input fusion function
            fused_seg = fuse_segmentations(load_segmentation(pred_file1),
                                               load_segmentation(pred_file2),
                                               filter_size=500,
                                               overlap_thresh=args.overlap_thresh,
                                               is_reset_label_ids=args.reset_label_ids)
            tifffile.imwrite(save_path, fused_seg.astype(np.int16))
        else:
            pred_file1 = os.path.join(res_path, args.pred_file1)
            pred_file2 = os.path.join(res_path, args.pred_file2)
            pred_file3 = os.path.join(res_path, args.pred_file3)
            fuse_segmentations_different_directions(pred_file1, pred_file2, pred_file3,
                                                    save_file=save_path,
                                                    overlap_thresh=args.overlap_thresh,
                                                    is_reset_label_ids=args.reset_label_ids)

