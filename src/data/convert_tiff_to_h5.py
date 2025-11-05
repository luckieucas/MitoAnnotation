#!/usr/bin/env python3
import argparse
import h5py
import tifffile as tiff
import numpy as np
from pathlib import Path
import os


def relabel_data(data: np.ndarray) -> np.ndarray:
    """
    Remap all non-zero labels in the input array to a consecutive range starting from 1.
    Returns the remapped array.
    """
    # Get unique labels excluding zero (background)
    unique_labels = np.unique(data)
    unique_labels = unique_labels[unique_labels != 0]
    print(f"Found {len(unique_labels)} unique non-zero labels: {unique_labels}")

    # Create a mapping from old label values to new consecutive labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

    # Initialize output array with the same shape and dtype as input
    remapped = np.zeros_like(data, dtype=data.dtype)

    # Apply the mapping to remap labels
    for old_label, new_label in label_mapping.items():
        remapped[data == old_label] = new_label

    return remapped


def convert_single_file(input_path: Path, output_path: Path, data_type: str, dtype: str, compression: str, compression_level: int):
    """
    Convert a single TIFF file to HDF5 format.
    """
    print(f"\nProcessing: {input_path}")
    
    # Read the TIFF file
    data = tiff.imread(str(input_path))
    print(f"  Data shape: {data.shape}")
    
    # Only relabel if the input type is segmentation
    if data_type == "seg":
        print("  Processing segmentation data - applying relabeling...")
        data = relabel_data(data)
        print(f"  Max label value after remapping: {np.max(data)}")
    else:
        print("  Processing image data - no relabeling applied")

    # Convert to desired data type if needed
    target_dtype = np.dtype(dtype)
    if data.dtype != target_dtype:
        data = data.astype(target_dtype)

    # Write the data to an HDF5 file
    print(f"  Saving to: {output_path}")
    with h5py.File(str(output_path), "w") as h5_file:
        h5_file.create_dataset(
            name="main",
            data=data,
            dtype=target_dtype,
            compression=compression,
            compression_opts=compression_level,
            chunks=True  # Let h5py choose an appropriate chunk size
        )
    
    print(f"  ✓ Completed: {output_path.name}")


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert TIFF file(s) (segmentation or image) to HDF5 format. "
                    "Supports single file or batch conversion from a directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input TIFF file or directory containing TIFF files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output HDF5 file (for single file) or directory (for batch conversion)"
    )
    parser.add_argument(
        "--type", choices=["seg", "img"], required=True,
        help="Type of input data: 'seg' for segmentation labels, 'img' for image data"
    )
    parser.add_argument(
        "--dtype", choices=["uint8", "uint16"], default="uint16",
        help="Data type for the output array (default: uint16)"
    )
    parser.add_argument(
        "--compression", type=str, default="gzip",
        help="Compression type for HDF5 dataset (default: gzip)"
    )
    parser.add_argument(
        "--compression-level", type=int, default=4,
        help="Compression level (0-9) for the chosen compression type (default: 4)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Single file conversion
        print("=" * 60)
        print("Single file conversion mode")
        print("=" * 60)
        convert_single_file(
            input_path=input_path,
            output_path=output_path,
            data_type=args.type,
            dtype=args.dtype,
            compression=args.compression,
            compression_level=args.compression_level
        )
        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
        
    elif input_path.is_dir():
        # Batch conversion from directory
        print("=" * 60)
        print("Batch conversion mode")
        print("=" * 60)
        
        # Find all TIFF files in the directory
        tiff_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        tiff_files = []
        for ext in tiff_extensions:
            tiff_files.extend(input_path.glob(ext))
        
        tiff_files = sorted(set(tiff_files))  # Remove duplicates and sort
        
        if not tiff_files:
            print(f"No TIFF files found in directory: {input_path}")
            return
        
        print(f"Found {len(tiff_files)} TIFF file(s) to convert")
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
        
        # Process each TIFF file
        success_count = 0
        fail_count = 0
        
        for i, tiff_file in enumerate(tiff_files, 1):
            try:
                print(f"\n[{i}/{len(tiff_files)}]", end=" ")
                
                # Generate output filename: replace .tif/.tiff with .h5
                output_filename = tiff_file.stem + ".h5"
                output_file = output_path / output_filename
                
                convert_single_file(
                    input_path=tiff_file,
                    output_path=output_file,
                    data_type=args.type,
                    dtype=args.dtype,
                    compression=args.compression,
                    compression_level=args.compression_level
                )
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {tiff_file.name}")
                print(f"    Error: {e}")
                fail_count += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch conversion complete!")
        print(f"  Successful: {success_count}/{len(tiff_files)}")
        if fail_count > 0:
            print(f"  Failed: {fail_count}/{len(tiff_files)}")
        print("=" * 60)
        
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return


if __name__ == "__main__":
    main()

