"""
Convert nnUNet dataset format to empanada format (2D slices).

This module provides utilities to convert 3D nnUNet datasets into 2D slice format
required by empanada/MitoNet training.
"""
import os
from pathlib import Path
from typing import Optional
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


def convert_nnunet_to_empanada_format(
    nnunet_dataset_path: Path,
    output_path: Path,
    min_foreground_pixels: int = 200,
    skip_empty_slices: bool = True,
) -> None:
    """
    Convert nnUNet dataset to empanada format (2D slices).

    Converts 3D NII.GZ files into 2D TIFF slices organized as:
        output_path/
            train/
                volume_name/
                    images/
                        slice_*.tif
                    masks/
                        slice_*.tif
            val/
                volume_name/
                    images/
                        slice_*.tif
                    masks/
                        slice_*.tif

    Args:
        nnunet_dataset_path: Path to nnUNet dataset directory.
        output_path: Output directory for converted data.
        min_foreground_pixels: Minimum number of foreground pixels to keep a slice.
        skip_empty_slices: Whether to skip slices with no foreground.
    """
    nnunet_dataset_path = Path(nnunet_dataset_path)
    output_path = Path(output_path)

    logger.info(f"Converting nnUNet dataset from {nnunet_dataset_path} to empanada format...")

    # Process training set
    logger.info("Processing training data...")
    images_tr_path = nnunet_dataset_path / "imagesTr"
    labels_tr_path = nnunet_dataset_path / "instancesTr"

    if not images_tr_path.exists():
        labels_tr_path = nnunet_dataset_path / "labelsTr"

    train_images = sorted(glob(str(images_tr_path / "*.nii.gz")))
    train_labels = sorted(glob(str(labels_tr_path / "*.nii.gz")))

    logger.info(f"Found {len(train_images)} training images and {len(train_labels)} training labels")

    _convert_volumes(
        train_images,
        train_labels,
        output_path / "train",
        min_foreground_pixels,
        skip_empty_slices,
    )

    # Process test/validation set
    logger.info("Processing test/validation data...")
    images_ts_path = nnunet_dataset_path / "imagesTs"
    instances_ts_path = nnunet_dataset_path / "instancesTs"

    if not instances_ts_path.exists():
        instances_ts_path = nnunet_dataset_path / "labelsTs"

    if images_ts_path.exists() and instances_ts_path.exists():
        test_images = sorted(glob(str(images_ts_path / "*.nii.gz")))
        test_labels = sorted(glob(str(instances_ts_path / "*.nii.gz")))

        logger.info(f"Found {len(test_images)} test images and {len(test_labels)} test labels")

        _convert_volumes(
            test_images,
            test_labels,
            output_path / "val",
            min_foreground_pixels,
            skip_empty_slices,
        )

    logger.info(f"Conversion complete. Output saved to {output_path}")


def _convert_volumes(
    image_files: list,
    label_files: list,
    output_dir: Path,
    min_foreground_pixels: int,
    skip_empty_slices: bool,
) -> None:
    """Convert a list of volumes to 2D slices."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_files, desc="Converting volumes"):
        img_path = Path(img_path)
        base_name = (
            img_path.name.replace("_0000", "")
            .replace(".nii.gz", "")
            .replace(".tiff", "")
            .replace(".tif", "")
        )

        # Find corresponding label file
        label_path = None
        for lbl_path in label_files:
            if base_name in Path(lbl_path).name:
                label_path = Path(lbl_path)
                break

        if label_path is None:
            logger.warning(f"No label found for {base_name}, skipping...")
            continue

        # Read 3D volumes
        try:
            img_sitk = sitk.ReadImage(str(img_path))
            img_3d = sitk.GetArrayFromImage(img_sitk)

            label_sitk = sitk.ReadImage(str(label_path))
            label_3d = sitk.GetArrayFromImage(label_sitk)
        except Exception as e:
            logger.error(f"Error reading {base_name}: {e}, skipping...")
            continue

        # Check shape compatibility
        if img_3d.shape != label_3d.shape:
            logger.warning(
                f"Shape mismatch for {base_name}: "
                f"img {img_3d.shape} vs label {label_3d.shape}, skipping..."
            )
            continue

        # Create volume directories
        volume_dir = output_dir / base_name
        images_dir = volume_dir / "images"
        masks_dir = volume_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Convert each slice
        num_slices = img_3d.shape[0]
        saved_slices = 0
        skipped_slices = 0

        for z in range(num_slices):
            slice_name = f"slice_{z:04d}.tif"
            label_slice = label_3d[z]

            # Skip empty slices if requested
            if skip_empty_slices:
                if np.sum(label_slice > 0) < min_foreground_pixels:
                    skipped_slices += 1
                    continue

            # Save image slice
            img_slice = img_3d[z]
            if img_slice.dtype != np.uint8:
                if img_slice.max() <= 255:
                    img_slice = img_slice.astype(np.uint8)
                else:
                    # Normalize to 0-255
                    img_min, img_max = img_slice.min(), img_slice.max()
                    if img_max > img_min:
                        img_slice = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img_slice = np.zeros_like(img_slice, dtype=np.uint8)

            tiff.imwrite(str(images_dir / slice_name), img_slice, compression="zlib")

            # Save label slice (ensure uint16 or uint32)
            label_slice = label_slice.astype(np.uint32)
            tiff.imwrite(str(masks_dir / slice_name), label_slice, compression="zlib")

            saved_slices += 1

        logger.debug(
            f"  {base_name}: Saved {saved_slices} slices, "
            f"Skipped {skipped_slices} empty slices"
        )
