#!/usr/bin/env python3
import argparse
import h5py
import tifffile as tiff
import numpy as np


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


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a TIFF file (segmentation or image) to HDF5 format"
    )
    parser.add_argument(
        "--input_tiff",
        help="Path to the input TIFF file"
    )
    parser.add_argument(
        "--output_h5",
        help="Path to the output HDF5 file"
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

    # Read the TIFF file
    print(f"Reading TIFF file: {args.input_tiff}")
    data = tiff.imread(args.input_tiff)
    print(f"Data shape: {data.shape}")
    
    # Only relabel if the input type is segmentation
    if args.type == "seg":
        print("Processing segmentation data - applying relabeling...")
        data = relabel_data(data)
        print(f"Max label value after remapping: {np.max(data)}")
    else:
        print("Processing image data - no relabeling applied")

    # Convert to desired data type if needed
    target_dtype = np.dtype(args.dtype)
    if data.dtype != target_dtype:
        data = data.astype(target_dtype)

    # Write the data to an HDF5 file
    print(f"Saving to HDF5 file: {args.output_h5}")
    with h5py.File(args.output_h5, "w") as h5_file:
        h5_file.create_dataset(
            name="main",
            data=data,
            dtype=target_dtype,
            compression=args.compression,
            compression_opts=args.compression_level,
            chunks=True  # Let h5py choose an appropriate chunk size
        )

    print("Conversion complete.")


if __name__ == "__main__":
    main()

