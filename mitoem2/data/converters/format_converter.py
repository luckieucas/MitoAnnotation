"""
General format conversion utilities.

Provides utilities for converting between different image and label formats.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile as tiff
import SimpleITK as sitk


def convert_image_format(
    input_path: Path,
    output_path: Path,
    output_format: str = "tiff",
    normalize: bool = True,
) -> None:
    """
    Convert image between different formats.

    Args:
        input_path: Path to input image.
        output_path: Path to output image.
        output_format: Output format (tiff, nii, nii.gz).
        normalize: Whether to normalize image to 0-255 range.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read input
    if input_path.suffix in [".nii", ".nii.gz"]:
        img_sitk = sitk.ReadImage(str(input_path))
        image = sitk.GetArrayFromImage(img_sitk)
    else:
        image = tiff.imread(str(input_path))

    # Normalize if needed
    if normalize and image.dtype != np.uint8:
        if image.max() <= 255:
            image = image.astype(np.uint8)
        else:
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in ["tiff", "tif"]:
        tiff.imwrite(str(output_path), image, compression="zlib")
    elif output_format in ["nii", "nii.gz"]:
        img_sitk = sitk.GetImageFromArray(image)
        sitk.WriteImage(img_sitk, str(output_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def convert_label_format(
    input_path: Path,
    output_path: Path,
    output_format: str = "tiff",
    dtype: type = np.uint32,
) -> None:
    """
    Convert label between different formats.

    Args:
        input_path: Path to input label.
        output_path: Path to output label.
        output_format: Output format (tiff, nii, nii.gz).
        dtype: Output data type.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read input
    if input_path.suffix in [".nii", ".nii.gz"]:
        label_sitk = sitk.ReadImage(str(input_path))
        label = sitk.GetArrayFromImage(label_sitk)
    else:
        label = tiff.imread(str(input_path))

    # Convert dtype
    label = label.astype(dtype)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in ["tiff", "tif"]:
        tiff.imwrite(str(output_path), label, compression="zlib")
    elif output_format in ["nii", "nii.gz"]:
        label_sitk = sitk.GetImageFromArray(label)
        sitk.WriteImage(label_sitk, str(output_path))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
