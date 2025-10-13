# -*- coding: utf-8 -*-
"""
Convert TIFF to NIfTI (.nii.gz) with user-specified spacing.

Usage:
  python tiff2nii.py --input path/to/input.tiff \
                     --output path/to/output.nii.gz \
                     --spacing 0.5 0.5 1.0
Notes:
  - spacing is given as (sx sy sz) in millimeters (or your unit).
  - If the input TIFF is 2D, the script will add a singleton z-dimension (depth=1).
  - If the input TIFF is 3D (multi-page TIFF), the script preserves its depth.
  - Assumes grayscale TIFF (single channel). Multi-channel (RGB) is not supported.
"""

import argparse
import SimpleITK as sitk
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input TIFF file path")
    p.add_argument("--output", required=True, help="Output NIfTI .nii.gz file path")
    p.add_argument("--spacing", nargs="+", type=float, required=True,
                   help="Voxel spacing: sx sy [sz]. If input is 2D and sz not given, defaults to 1.0")
    p.add_argument("--dtype", default=None,
                   help="Optional output dtype, e.g., uint8, int16, float32. Default: keep original.")
    return p.parse_args()

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

    # Read TIFF (SimpleITK supports single- or multi-page TIFF)
    img = sitk.ReadImage(args.input)

    # Disallow vector (RGB) images (common in natural images)
    if img.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("This script expects a grayscale TIFF (1 component per pixel).")

    # Make sure we output 3D NIfTI (2D TIFF will be expanded to depth=1)
    img = ensure_3d(img)

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

    # Apply spacing, preserve origin and direction (if none, set defaults)
    # For a clean header, we can keep origin and direction as-is; if empty, set to identity.
    origin = img.GetOrigin() if img.GetOrigin() else (0.0, 0.0, 0.0)
    direction = img.GetDirection()
    if not direction or len(direction) != 9:
        direction = tuple(np.eye(3).ravel().tolist())

    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)

    # Optional dtype cast
    if args.dtype is not None:
        np2sitk = {
            "uint8": sitk.sitkUInt8,
            "int16": sitk.sitkInt16,
            "uint16": sitk.sitkUInt16,
            "int32": sitk.sitkInt32,
            "uint32": sitk.sitkUInt32,
            "float32": sitk.sitkFloat32,
            "float64": sitk.sitkFloat64,
        }
        pixel_id = np2sitk.get(args.dtype.lower())
        if pixel_id is None:
            raise ValueError(f"Unsupported dtype: {args.dtype}")
        img = sitk.Cast(img, pixel_id)

    # Write NIfTI (.nii.gz)
    sitk.WriteImage(img, args.output, useCompression=True)
    print(f"Saved: {args.output}")
    print(f"Size: {img.GetSize()}  Spacing: {img.GetSpacing()}  Origin: {img.GetOrigin()}")

if __name__ == "__main__":
    main()