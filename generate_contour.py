"""
Generate instance contour maps from segmentation masks.
Supports both 3D TIFF and NIfTI (.nii/.nii.gz) inputs.
"""
import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import tifffile as tiff
import nibabel as nib
from tqdm import tqdm

from connectomics.data.utils.data_segmentation import seg_to_instance_bd


def read_volume(path):
    """Read a 3D mask volume from TIFF or NIfTI."""
    p = Path(path)
    suffixes = p.suffixes
    if suffixes[-2:] == ['.nii', '.gz']:
        img = nib.load(path)
        vol = img.get_fdata().astype(np.uint16)
        affine = img.affine
    elif suffixes[-1].lower() in ['.nii']:
        img = nib.load(path)
        vol = img.get_fdata().astype(np.uint16)
        affine = img.affine
    elif suffixes[-1].lower() in ('.tif', '.tiff'):
        vol = tiff.imread(path).astype(np.uint16)
        affine = None
    else:
        raise ValueError(f"Unsupported file extension: {''.join(suffixes)}")
    return vol, affine


def write_volume(vol, affine, out_path):
    """Write the contour volume back to TIFF or NIfTI, matching extension."""
    p = Path(out_path)
    suffixes = p.suffixes
    if suffixes[-2:] == ['.nii', '.gz'] or suffixes[-1].lower() == '.nii':
        nib.save(nib.Nifti1Image(vol.astype(np.uint8), affine), out_path)
    elif suffixes[-1].lower() in ('.tif', '.tiff'):
        tiff.imwrite(out_path, vol.astype(np.uint8),compression="zlib")
    else:
        raise ValueError(f"Unsupported output extension: {''.join(suffixes)}")


def generate_contour_map(mask):
    """
    Given a 3D label mask array, compute instance contour map.
    Returns a uint8 volume where boundary voxels == 1.
    """
    binary = (mask > 0).astype(np.uint8)
    contour = seg_to_instance_bd(binary, tzh=3)
    return contour.astype(np.uint8)


def process_file(input_path, output_folder):
    vol, affine = read_volume(input_path)
    print(f"Unique labels in volume: {np.unique(vol)}")
    contour = generate_contour_map(vol)
    contour[contour > 0] = 2
    binary = (vol > 0).astype(np.uint8)
    saved_mask = binary + contour
    saved_mask[saved_mask > 2] = 1

    p = Path(input_path)
    suffixes = p.suffixes
    # determine base name without all suffixes
    if suffixes[-2:] == ['.nii', '.gz']:
        base = p.name[:-7]
        ext = '.nii.gz'
    else:
        base = p.stem
        ext = suffixes[-1]

    out_name = f"{base}{ext}"
    out_path = os.path.join(output_folder, out_name)

    write_volume(saved_mask, affine, out_path)
    print(f"Saved contour: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-generate instance contour maps from 3D TIFF or NIfTI masks."
    )
    parser.add_argument("-i", "--input_folder", required=True,
                        help="Folder containing .tif/.tiff and/or .nii/.nii.gz mask files")
    parser.add_argument("-o", "--output_folder", required=True,
                        help="Folder to save contour maps")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    patterns = ["/*.tif", "/*.tiff", "/*.nii", "/*.nii.gz"]
    files = []
    for pat in patterns:
        files.extend(glob(args.input_folder + pat))
    for path in tqdm(files, desc="Processing masks"):
        try:
            process_file(path, args.output_folder)
        except Exception as e:
            print(f"[Error] {path}: {e}")
