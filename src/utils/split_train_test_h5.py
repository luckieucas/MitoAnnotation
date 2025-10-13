#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split full mitochondria dataset into train and test sets.
This script:
1. Creates mito_train.h5 from mito_full.h5 by removing regions in mito_test.h5
2. Creates im_train.h5 by masking out mito_test regions from im.h5
3. Creates im_test.h5 by masking out mito_train regions from im.h5
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def read_h5(file_path, dataset_name='main'):
    """
    Read data from h5 file.
    
    Args:
        file_path: Path to h5 file
        dataset_name: Name of dataset in h5 file (default: 'main')
    
    Returns:
        numpy array with the data
    """
    with h5py.File(file_path, 'r') as f:
        # Try to find the dataset
        if dataset_name in f:
            data = f[dataset_name][:]
        else:
            # If 'main' doesn't exist, try to use the first dataset
            keys = list(f.keys())
            if len(keys) > 0:
                print(f"Dataset '{dataset_name}' not found. Using '{keys[0]}' instead.")
                data = f[keys[0]][:]
            else:
                raise ValueError(f"No datasets found in {file_path}")
    return data


def write_h5(file_path, data, dataset_name='main', compression='gzip'):
    """
    Write data to h5 file.
    
    Args:
        file_path: Path to output h5 file
        data: numpy array to save
        dataset_name: Name of dataset in h5 file (default: 'main')
        compression: Compression type (default: 'gzip')
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, data=data, compression=compression)
    print(f"Saved: {file_path}")


def split_train_test(im_path, mito_full_path, mito_test_path, 
                     output_dir=None, dataset_name='main'):
    """
    Split mitochondria data into train and test sets.
    
    Args:
        im_path: Path to im.h5
        mito_full_path: Path to mito_full.h5
        mito_test_path: Path to mito_test.h5
        output_dir: Output directory (default: same as input files)
        dataset_name: Name of dataset in h5 files
    """
    # Convert paths to Path objects
    im_path = Path(im_path)
    mito_full_path = Path(mito_full_path)
    mito_test_path = Path(mito_test_path)
    
    if output_dir is None:
        output_dir = im_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Reading input files...")
    print(f"  - {im_path}")
    im = read_h5(im_path, dataset_name)
    print(f"    Shape: {im.shape}, dtype: {im.dtype}")
    
    print(f"  - {mito_full_path}")
    mito_full = read_h5(mito_full_path, dataset_name)
    print(f"    Shape: {mito_full.shape}, dtype: {mito_full.dtype}")
    
    print(f"  - {mito_test_path}")
    mito_test = read_h5(mito_test_path, dataset_name)
    print(f"    Shape: {mito_test.shape}, dtype: {mito_test.dtype}")
    
    # Verify shapes match
    if im.shape != mito_full.shape or im.shape != mito_test.shape:
        print("Warning: Input shapes don't match!")
        print(f"  im: {im.shape}, mito_full: {mito_full.shape}, mito_test: {mito_test.shape}")
    
    # Step 1: Create mito_train.h5
    # mito_train contains mitochondria from mito_full that are NOT in mito_test
    print("\n1. Creating mito_train.h5...")
    print("   Removing mito_test regions from mito_full...")
    
    # Create mask for test regions (where mito_test > 0)
    test_mask = mito_test > 0
    
    # Copy mito_full and set test regions to 0
    mito_train = mito_full.copy()
    mito_train[test_mask] = 0
    
    n_test_voxels = np.sum(test_mask)
    n_train_voxels = np.sum(mito_train > 0)
    n_full_voxels = np.sum(mito_full > 0)
    
    print(f"   Full mito voxels: {n_full_voxels}")
    print(f"   Test mito voxels: {n_test_voxels}")
    print(f"   Train mito voxels: {n_train_voxels}")
    print(f"   Verification: train + test = {n_train_voxels + n_test_voxels} "
          f"(should be close to {n_full_voxels})")
    
    mito_train_path = output_dir / "mito_train.h5"
    write_h5(mito_train_path, mito_train, dataset_name)
    
    # Step 2: Create im_train.h5
    # im_train is im with mito_test regions set to 0
    print("\n2. Creating im_train.h5...")
    print("   Masking mito_test regions in im...")
    
    im_train = im.copy()
    im_train[test_mask] = 0
    
    im_train_path = output_dir / "im_train.h5"
    write_h5(im_train_path, im_train, dataset_name)
    
    # Step 3: Create im_test.h5
    # im_test is im with mito_train regions set to 0
    print("\n3. Creating im_test.h5...")
    print("   Masking mito_train regions in im...")
    
    train_mask = mito_train > 0
    im_test = im.copy()
    im_test[train_mask] = 0
    
    im_test_path = output_dir / "im_test.h5"
    write_h5(im_test_path, im_test, dataset_name)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print(f"Output files saved to: {output_dir}")
    print(f"  - mito_train.h5")
    print(f"  - im_train.h5")
    print(f"  - im_test.h5")
    print()
    print("Summary:")
    print(f"  Full mitochondria voxels: {n_full_voxels:,}")
    print(f"  Test mitochondria voxels: {n_test_voxels:,}")
    print(f"  Train mitochondria voxels: {n_train_voxels:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Split mitochondria dataset into train and test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python split_train_test_h5.py im.h5 mito_full.h5 mito_test.h5
  python split_train_test_h5.py im.h5 mito_full.h5 mito_test.h5 --output-dir ./output
  python split_train_test_h5.py im.h5 mito_full.h5 mito_test.h5 --dataset-name volumes/raw
        """
    )
    
    parser.add_argument('im_path', type=str,
                        help='Path to im.h5 (input image)')
    parser.add_argument('mito_full_path', type=str,
                        help='Path to mito_full.h5 (full mitochondria labels)')
    parser.add_argument('mito_test_path', type=str,
                        help='Path to mito_test.h5 (test mitochondria labels)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as input files)')
    parser.add_argument('--dataset-name', type=str, default='main',
                        help='Name of dataset in h5 files (default: main)')
    
    args = parser.parse_args()
    
    split_train_test(
        args.im_path,
        args.mito_full_path,
        args.mito_test_path,
        args.output_dir,
        args.dataset_name
    )


if __name__ == '__main__':
    main()



