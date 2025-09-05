import argparse
import tifffile as tiff
import numpy as np
from scipy.ndimage import zoom
import os

def resize_tiff(input_path, output_path, target_size, mode):
    """
    Resize a 3D TIFF image or mask.

    Args:
        input_path (str): Path to the input TIFF file.
        output_path (str): Path to save the resized TIFF file.
        target_size (tuple): Target size (D, H, W).
        mode (str): "image" for linear interpolation, "mask" for nearest neighbor.

    Returns:
        None
    """
    # Read the input TIFF file
    volume = tiff.imread(input_path)
    print(f"Original shape: {volume.shape}, Target shape: {target_size}")
    
    # Compute zoom factors based on the target size and the current volume shape
    zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]

    # Choose interpolation method based on the mode
    if mode == "image":
        resized_volume = zoom(volume, zoom_factors, order=1)  # Linear interpolation
    elif mode == "mask":
        resized_volume = zoom(volume, zoom_factors, order=0)  # Nearest-neighbor interpolation
    else:
        raise ValueError("Mode must be either 'image' or 'mask'")

    # Save the resized volume with the same data type as the original volume
    tiff.imwrite(output_path, resized_volume.astype(volume.dtype), compression="zlib")
    print(f"Resized {mode} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a 3D TIFF image or mask.")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to the input TIFF file or folder.")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the resized TIFF file or folder.")
    parser.add_argument("--size", type=int, nargs=3, required=True, 
                        help="Target size (D, H, W).")
    parser.add_argument("--mode", type=str, choices=["image", "mask"], required=True, 
                        help="Resize mode: 'image' for linear interpolation, 'mask' for nearest neighbor.")

    args = parser.parse_args()

    # Check if the input path is a directory
    if os.path.isdir(args.input):
        # Ensure the output folder exists; create it if necessary
        os.makedirs(args.output, exist_ok=True)
        # Iterate over each file in the input folder
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('0000.tif', '0000.tiff')):
                input_file = os.path.join(args.input, filename)
                output_file = os.path.join(args.output, filename)
                print(f"Processing {input_file} ...")
                resize_tiff(input_file, output_file, tuple(args.size), args.mode)
    else:
        # Process a single file if input is not a directory
        resize_tiff(args.input, args.output, tuple(args.size), args.mode)
