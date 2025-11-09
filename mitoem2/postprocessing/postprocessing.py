# process_mask_to_shape.py

import argparse
import tifffile
import numpy as np
from pathlib import Path
from skimage import measure
from scipy.ndimage import zoom

def process_3d_mask(mask_path: Path, image_path: Path, min_size: int, compression: str) -> None:
    """
    Processes a 3D TIFF mask by filtering, relabeling, and upsampling to match a reference image's shape.

    Args:
        mask_path (Path): The path to the input 3D TIFF mask file.
        image_path (Path): The path to the reference image TIFF file to match the shape of.
        min_size (int): The minimum size (voxel count) for a label to be kept.
        compression (str): The compression method for the output TIFF. 'none' for no compression.
    """
    # --- 1. Load the input MASK file ---
    print(f"Reading input mask file: {mask_path}")
    try:
        mask_data = tifffile.imread(mask_path)
        print(f"Original mask dimensions: {mask_data.shape}")
        print(f"Original number of labels (including background): {len(np.unique(mask_data))}")
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return
    except Exception as e:
        print(f"Error reading mask TIFF file: {e}")
        return

    # --- 2. Filter out labels with a size smaller than min_size ---
    print(f"Filtering labels smaller than {min_size} voxels...")
    filtered_mask = mask_data.copy()
    unique_labels, counts = np.unique(filtered_mask, return_counts=True)
    small_labels = unique_labels[(counts < min_size) & (unique_labels != 0)]
    
    if small_labels.size > 0:
        print(f"Removing {len(small_labels)} small labels.")
        filtered_mask[np.isin(filtered_mask, small_labels)] = 0
    else:
        print("No small labels to remove.")

    # --- 3. Relabel all remaining objects to be consecutive from 1 ---
    print("Relabeling to ensure consecutive numbering...")
    relabeled_mask, num_labels = measure.label(filtered_mask, background=0, return_num=True)
    print(f"New number of labels (excluding background): {num_labels}")

    # --- 4. Upsample the mask to match the reference image shape ---
    # 4a. Load reference image to get target shape
    print(f"Reading reference image file to get target shape: {image_path}")
    try:
        target_shape = tifffile.imread(image_path).shape
        print(f"Target shape from reference image: {target_shape}")
    except FileNotFoundError:
        print(f"Error: Reference image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error reading reference TIFF file: {e}")
        return
    
    current_mask_shape = relabeled_mask.shape
    
    if current_mask_shape == target_shape:
        print("Mask already has the target shape. No upsampling needed.")
        upsampled_mask = relabeled_mask
    else:
        print(f"Upsampling mask from {current_mask_shape} to {target_shape}...")
        
        # 4b. Calculate the zoom factor for each dimension. Must be floating point division.
        zoom_factors = np.array(target_shape) / np.array(current_mask_shape)
        print(f"Calculated zoom factors: {zoom_factors}")
    
        # 4c. Use scipy.ndimage.zoom with the calculated factors.
        # order=0 is critical for nearest-neighbor interpolation to preserve integer labels.
        upsampled_mask = zoom(relabeled_mask, zoom_factors, order=0, output=relabeled_mask.dtype)
        
        # Due to floating point precision, the result might be off by one pixel.
        # This check ensures the final output is exactly the target shape by cropping if necessary.
        if upsampled_mask.shape != target_shape:
            print(f"Warning: Zoom result shape {upsampled_mask.shape} differs from target. Cropping to match.")
            slices = tuple(slice(0, s) for s in target_shape)
            upsampled_mask = upsampled_mask[slices]

        print(f"Final upsampled mask dimensions: {upsampled_mask.shape}")

    # --- 5. Save the final processed mask ---
    output_path = mask_path.with_name(f"{mask_path.stem}_upsampled.tiff")
    
    compression_args = {}
    if compression and compression.lower() != 'none':
        compression_args['compression'] = compression
        print(f"Saving processed mask to: {output_path} with '{compression}' compression.")
    else:
        print(f"Saving processed mask to: {output_path} with no compression.")

    try:
        tifffile.imwrite(output_path, upsampled_mask.astype(np.uint16), **compression_args)
        print("Processing complete.")
    except Exception as e:
        print(f"Error writing output file: {e}")

def main():
    """
    Main function to parse command-line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description='Filter, relabel, and upsample a 3D TIFF mask to match a reference image shape.'
    )
    parser.add_argument(
        '--mask_file',
        type=str,
        help='Path to the input 3D TIFF mask file to be processed.'
    )
    parser.add_argument(
        '--image_file',
        type=str,
        required=True, # This argument is now mandatory
        help='Path to the reference image TIFF file to match the shape of.'
    )
    parser.add_argument(
        '--min_size',
        type=int,
        default=600,
        help='The minimum size (voxel count) for a labeled region to be kept. Default is 600.'
    )
    parser.add_argument(
        '--compression',
        type=str,
        default='zlib',
        choices=['zlib', 'packbits', 'lzw', 'none'],
        help="Compression for the output TIFF. 'zlib' is a good default. Default is 'zlib'."
    )

    args = parser.parse_args()
    mask_path = Path(args.mask_file)
    image_path = Path(args.image_file)
    
    process_3d_mask(mask_path, image_path, args.min_size, args.compression)

if __name__ == '__main__':
    main()