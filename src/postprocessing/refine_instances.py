#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refine 3-D instance segmentation by combining an accurate binary mask with a
partial/coarse instance mask.

Usage
-----
python refine_instances.py --binary binary.nii.gz \
                           --instances coarse.tiff \
                           --output refined.nii.gz
"""
from typing import Union, Tuple
import argparse
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import ball, remove_small_objects
import tifffile as tiff
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom

# ---------- I/O helpers ----------------------------------------------------- #
def read_volume(path: Path) -> Tuple[np.ndarray, Union[sitk.Image, None]]:
    """Read a 3-D volume from .tiff / .nii / .nii.gz."""
    suffixes = path.suffixes
    if suffixes[-2:] == [".nii", ".gz"] or suffixes[-1] == ".nii":
        img = nib.load(str(path))
        vol = img.get_fdata().astype(np.uint16)
        return vol, img.affine
    elif suffixes[-1].lower() in (".tif", ".tiff"):
        vol = tiff.imread(str(path)).astype(np.uint16)
        return vol, None
    else:
        raise ValueError(f"Unsupported file type: {path}")


def save_volume(vol: np.ndarray, ref_img: Union[sitk.Image, None], path: Path) -> None:
    """Save a 3-D volume to .tiff / .nii.gz (determined by extension)."""
    if path.suffixes[-2:] == [".nii", ".gz"] or path.suffix == ".nii":
        img = nib.Nifti1Image(vol.astype(np.uint16), affine if affine is not None else np.eye(4))
        nib.save(img, str(path))
    elif path.suffix.lower() in (".tif", ".tiff"):
        tiff.imwrite(str(path), vol.astype(np.uint16), compression="zlib")
    else:
        raise ValueError(f"Unsupported output type: {path}")


# ---------- Core algorithm -------------------------------------------------- #
def refine_instance_seg(
    binary_mask: np.ndarray,
    coarse_instances: np.ndarray,
    min_seed_vox: int = 50,
    seed_dilate_iter: int = 1,
    min_final_vox: int = 200,
) -> np.ndarray:
    """Combine binary and coarse masks to obtain refined instances."""
    # 1) Clean small seeds
    seeds_clean = remove_small_objects(coarse_instances, min_size=min_seed_vox)

    # 2) Optional dilation to fill tiny holes inside seeds
    if seed_dilate_iter > 0:
        structure = ball(1)
        for _ in range(seed_dilate_iter):
            seeds_clean = ndi.grey_dilation(seeds_clean, footprint=structure)

    # 3) Distance transform inside binary mask
    distance = ndi.distance_transform_edt(binary_mask.astype(bool))

    # 4) Marker-controlled watershed (negative distance => basins interior)
    refined = watershed(-distance, markers=seeds_clean, mask=binary_mask.astype(bool))

    # 5) Remove tiny final instances
    refined = remove_small_objects(refined, min_size=min_final_vox)

    # 6) Relabel instances to be 1..N (background=0)
    unique_vals = np.unique(refined)
    unique_vals = unique_vals[unique_vals != 0]  # exclude background
    relabeled = np.zeros_like(refined, dtype=np.uint16)
    for new_id, old_id in enumerate(unique_vals, start=1):
        relabeled[refined == old_id] = new_id

    return relabeled


# ---------- CLI ------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine 3-D instance segmentation using a binary mask and a coarse instance mask."
    )
    p.add_argument("--binary", required=True, type=Path, help="Binary mask file (.tiff / .nii / .nii.gz)")
    p.add_argument("--instances", required=True, type=Path, help="Coarse instance mask file")
    p.add_argument("--output", required=True, type=Path, help="Output file path")
    p.add_argument("--min_seed_vox", type=int, default=50, help="Minimum voxels to keep a seed label")
    p.add_argument("--seed_dilate_iter", type=int, default=1, help="Number of dilations applied to seeds")
    p.add_argument("--min_final_vox", type=int, default=500, help="Remove final labels smaller than this")
    return p.parse_args()


def main() -> None:
    args = get_args()

    # Load volumes
    binary, affine_bin = read_volume(args.binary)
    binary[binary > 1] = 0  # Ensure binary mask is boolean
    #binary = binary[::2, ::2, ::2]  # Downsample by factor of 2

    # Load coarse instance mask
    seeds, _ = read_volume(args.instances)
    
    if seeds.shape != binary.shape:
        print(f"Resizing seeds to binary shape: {seeds.shape} -> {binary.shape}")
        # resize seeds to binary shape
        zoom_factors = [t / s for t, s in zip(binary.shape, seeds.shape)]
        seeds = zoom(seeds, zoom_factors, order=0)
    
    #seeds = seeds[::2, ::2, ::2]  # Downsample by factor of 2
    if binary.shape != seeds.shape:
        raise ValueError("Binary mask and instance mask must have identical shapes.")

    # Run refinement
    refined = refine_instance_seg(
        binary_mask=binary,
        coarse_instances=seeds,
        min_seed_vox=args.min_seed_vox,
        seed_dilate_iter=args.seed_dilate_iter,
        min_final_vox=args.min_final_vox,
    )

    # Save result (use binary's affine if writing NIfTI)
    save_volume(refined, affine_bin, args.output)
    print(f"[✓] Refined instance segmentation saved to: {args.output}")


if __name__ == "__main__":
    main()
