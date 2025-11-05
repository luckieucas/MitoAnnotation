# -*- coding: utf-8 -*-
"""
Convert TIFF to NIfTI (.nii.gz) with user-specified spacing.

Usage:
  # Single file conversion:
  python tiff2nii.py --input path/to/input.tiff \
                     --output path/to/output.nii.gz \
                     --spacing 0.5 0.5 1.0
  
  # Single file conversion (output auto-generated):
  python tiff2nii.py --input path/to/input.tiff \
                     --spacing 0.5 0.5 1.0
  
  # Folder batch conversion:
  python tiff2nii.py --input path/to/input_folder \
                     --output path/to/output_folder \
                     --spacing 0.5 0.5 1.0
  
  # Folder batch conversion (output uses input path):
  python tiff2nii.py --input path/to/input_folder \
                     --spacing 0.5 0.5 1.0

Notes:
  - spacing is given as (sx sy sz) in millimeters (or your unit).
  - If the input TIFF is 2D, the script will add a singleton z-dimension (depth=1).
  - If the input TIFF is 3D (multi-page TIFF), the script preserves its depth.
  - Assumes grayscale TIFF (single channel). Multi-channel (RGB) is not supported.
  - Supports both single file and batch folder processing.
  - When input is a folder, recursively processes all TIFF files in all subdirectories.
  - Automatically skips directories named "2d_slices" during recursive search.
  - Output files preserve the directory structure of input files.
  - If --output is not specified, output will be generated based on input path:
    * For a file: output.nii.gz will be in the same directory as input
    * For a folder: output files will be placed in the same folder as input files (preserving subdirectory structure)
"""

import argparse
import os
import glob
import SimpleITK as sitk
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, 
                   help="Input TIFF file path or folder containing TIFF files. If folder, recursively processes all TIFF files in all subdirectories.")
    p.add_argument("--output", default=None, 
                   help="Output NIfTI .nii.gz file path or folder for batch conversion. If not specified, output will be generated based on input path. Directory structure is preserved for folder input.")
    p.add_argument("--spacing", nargs="+", type=float, required=True,
                   help="Voxel spacing: sx sy [sz]. If input is 2D and sz not given, defaults to 1.0")
    p.add_argument("--dtype", default=None,
                   help="Optional output dtype, e.g., uint8, int16, float32. Default: keep original.")
    return p.parse_args()

def find_tiff_files(folder_path, exclude_dirs=None):
    """
    Find all TIFF files in a folder recursively (including all subdirectories).
    
    Args:
        folder_path: Root folder path to search
        exclude_dirs: List of directory names to exclude (default: ['2d_slices'])
    
    Returns:
        Sorted list of absolute paths to all TIFF files found
    """
    if exclude_dirs is None:
        exclude_dirs = ['2d_slices']
    
    tiff_extensions = ['*.tiff', '*.tif', '*.TIFF', '*.TIF']
    tiff_files = []
    for ext in tiff_extensions:
        # Recursively search for TIFF files in all subdirectories
        pattern = os.path.join(folder_path, '**', ext)
        found_files = glob.glob(pattern, recursive=True)
        # Filter out files in excluded directories
        for file_path in found_files:
            # Check if any excluded directory is in the file path
            should_exclude = False
            # Normalize path separators for consistent checking
            normalized_path = file_path.replace(os.path.altsep, os.path.sep) if os.path.altsep else file_path
            for exclude_dir in exclude_dirs:
                # Check if path contains the excluded directory as a directory (with path separators)
                # Pattern: /dirname/ or path starts with dirname/
                if ((os.path.sep + exclude_dir + os.path.sep) in normalized_path or
                    normalized_path.startswith(exclude_dir + os.path.sep)):
                    should_exclude = True
                    break
            if not should_exclude:
                tiff_files.append(file_path)
    
    # Remove duplicates and sort
    tiff_files = sorted(list(set(tiff_files)))
    return tiff_files

def convert_tiff_to_nii(input_path, output_path, spacing, dtype=None):
    """
    Convert a single TIFF file to NIfTI format.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to output NIfTI file
        spacing: Tuple of (sx, sy, sz) spacing values
        dtype: Optional output data type
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Read TIFF (SimpleITK supports single- or multi-page TIFF)
        img = sitk.ReadImage(input_path)

        # Disallow vector (RGB) images (common in natural images)
        if img.GetNumberOfComponentsPerPixel() != 1:
            return False, f"Error: {input_path} is not a grayscale TIFF (1 component per pixel)."

        # Make sure we output 3D NIfTI (2D TIFF will be expanded to depth=1)
        img = ensure_3d(img)

        # Apply spacing, preserve origin and direction (if none, set defaults)
        origin = img.GetOrigin() if img.GetOrigin() else (0.0, 0.0, 0.0)
        direction = img.GetDirection()
        if not direction or len(direction) != 9:
            direction = tuple(np.eye(3).ravel().tolist())

        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)

        # Optional dtype cast
        if dtype is not None:
            np2sitk = {
                "uint8": sitk.sitkUInt8,
                "int16": sitk.sitkInt16,
                "uint16": sitk.sitkUInt16,
                "int32": sitk.sitkInt32,
                "uint32": sitk.sitkUInt32,
                "float32": sitk.sitkFloat32,
                "float64": sitk.sitkFloat64,
            }
            pixel_id = np2sitk.get(dtype.lower())
            if pixel_id is None:
                return False, f"Unsupported dtype: {dtype}"
            img = sitk.Cast(img, pixel_id)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Write NIfTI (.nii.gz)
        sitk.WriteImage(img, output_path, useCompression=True)
        
        return True, f"Saved: {output_path} (Size: {img.GetSize()}, Spacing: {img.GetSpacing()})"
    
    except Exception as e:
        return False, f"Error processing {input_path}: {str(e)}"

def ensure_3d(img):
    """If image is 2D, convert to 3D by adding a singleton z dimension."""
    if img.GetDimension() == 2:
        # JoinSeries turns a 2D image into a 3D image with depth=1
        img3d = sitk.JoinSeries([img])
        # Fix direction to identity for 3D (3x3) because JoinSeries may set 2D direction extended
        img3d.SetDirection(tuple(np.eye(3).ravel().tolist()))
        return img3d
    elif img.GetDimension() == 3:
        return img
    else:
        raise ValueError(f"Only 2D/3D supported, got dimension={img.GetDimension()}")

def main():
    args = parse_args()

    # Determine spacing
    # If user provided 2 values and image is 3D (after ensure_3d), infer sz=1.0
    if len(args.spacing) not in (2, 3):
        raise ValueError("Spacing must have 2 or 3 numbers: sx sy [sz].")
    if len(args.spacing) == 2:
        sx, sy = args.spacing
        sz = 1.0
    else:
        sx, sy, sz = args.spacing
    spacing = (float(sx), float(sy), float(sz))

    # Check if input is a file or folder
    input_is_dir = os.path.isdir(args.input)
    input_is_file = os.path.isfile(args.input)
    
    if not input_is_file and not input_is_dir:
        raise ValueError(f"Input path does not exist: {args.input}")

    # Auto-generate output path if not specified
    if args.output is None:
        if input_is_dir:
            # For folder input, use the same folder as output
            args.output = args.input
        else:
            # For file input, generate output filename in the same directory
            input_dir = os.path.dirname(args.input)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            # Handle .tiff extension (has two dots)
            if base_name.endswith('.tif'):
                base_name = base_name[:-4]
            args.output = os.path.join(input_dir, base_name + '.nii.gz') if input_dir else (base_name + '.nii.gz')

    # Validate input/output combinations
    if input_is_dir:
        # If input is a folder, output must be a folder
        # Check if output path exists and is a file
        if os.path.isfile(args.output):
            raise ValueError(f"If input is a folder, output must be a folder. Got existing file: {args.output}")
        
        # Create output folder if it doesn't exist (and it's not the same as input)
        if args.output != args.input:
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            elif not os.path.isdir(args.output):
                raise ValueError(f"Output path exists but is not a directory: {args.output}")
        
        # Find all TIFF files in input folder
        tiff_files = find_tiff_files(args.input)
        
        if not tiff_files:
            print(f"Warning: No TIFF files found in {args.input} (including subdirectories)")
            return
        
        print(f"Found {len(tiff_files)} TIFF file(s) to convert (including all subdirectories)...")
        
        # Process each TIFF file
        success_count = 0
        error_count = 0
        
        for tiff_file in tiff_files:
            # Get relative path from input folder to preserve subdirectory structure
            rel_path = os.path.relpath(tiff_file, args.input)
            
            # Generate output filename: replace extension with .nii.gz
            base_name = os.path.splitext(rel_path)[0]
            # Handle .tiff extension (has two dots)
            if base_name.endswith('.tif'):
                base_name = base_name[:-4]
            
            output_file = os.path.join(args.output, base_name + '.nii.gz')
            
            # Convert file
            success, message = convert_tiff_to_nii(tiff_file, output_file, spacing, args.dtype)
            
            if success:
                # Show relative path to indicate subdirectory structure
                print(f"[{success_count + 1}/{len(tiff_files)}] {rel_path} -> {message}")
                success_count += 1
            else:
                print(f"[ERROR] {rel_path}: {message}")
                error_count += 1
        
        print(f"\nConversion complete: {success_count} succeeded, {error_count} failed")
        
    else:
        # Single file conversion
        # Check if output is a directory or file
        if os.path.isdir(args.output):
            # If output is an existing folder, generate filename based on input filename
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            # Handle .tiff extension (has two dots)
            if base_name.endswith('.tif'):
                base_name = base_name[:-4]
            
            output_file = os.path.join(args.output, base_name + '.nii.gz')
        elif args.output.endswith(os.sep) or (not os.path.exists(args.output) and not args.output.endswith('.nii.gz')):
            # If output path ends with separator or doesn't exist and doesn't look like a file, treat as folder
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            if base_name.endswith('.tif'):
                base_name = base_name[:-4]
            
            output_file = os.path.join(args.output, base_name + '.nii.gz')
        else:
            # Output is a file path
            output_file = args.output
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        # Convert single file
        success, message = convert_tiff_to_nii(args.input, output_file, spacing, args.dtype)
        
        if success:
            print(message)
        else:
            print(f"ERROR: {message}")
            exit(1)

if __name__ == "__main__":
    main()