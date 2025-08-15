#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from tifffile import imread, imwrite
from skimage.morphology import skeletonize
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from connectomics.data.utils.data_transform import skeleton_aware_distance_transform


def process_slice(z, label_2d, bg_value, relabel, padding,
                  resolution_xy, alpha, smooth, smooth_skeleton_only):
    """Worker function to process one slice and return skeleton mask."""
    _, semantic = skeleton_aware_distance_transform(
        label=label_2d,
        bg_value=bg_value,
        relabel=relabel,
        padding=padding,
        resolution=tuple(resolution_xy),
        alpha=alpha,
        smooth=smooth,
        smooth_skeleton_only=smooth_skeleton_only,
    )
    sk2d = skeletonize((semantic > 0).astype(bool)).astype(np.uint8)
    return z, sk2d


def run_slice_skeleton_parallel(
    input_tiff: str,
    output_tiff: str,
    resolution_xy=(1.0, 1.0),
    alpha: float = 0.8,
    relabel: bool = True,
    padding: bool = False,
    smooth: bool = True,
    smooth_skeleton_only: bool = True,
    bg_value: float = -1.0,
    num_workers: int = None,
):
    vol = imread(input_tiff)
    if vol.ndim != 3:
        raise ValueError(f"Input must be 3D (Z,Y,X). Got shape {vol.shape}")
    if not np.issubdtype(vol.dtype, np.integer):
        vol = vol.astype(np.int32)

    Z, Y, X = vol.shape
    sk_stack = np.zeros((Z, Y, X), dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_slice,
                z, vol[z], bg_value, relabel, padding,
                resolution_xy, alpha, smooth, smooth_skeleton_only
            )
            for z in range(Z)
        ]

        # tqdm progress bar
        for f in tqdm(as_completed(futures), total=Z, desc="Processing slices"):
            z, sk2d = f.result()
            sk_stack[z] = sk2d

    imwrite(output_tiff, sk_stack, dtype=np.uint8, compression="zlib")
    print(f"Saved skeleton stack to: {output_tiff}")


def parse_args():
    p = argparse.ArgumentParser(description="Parallel slice-wise skeleton from 3D TIFF with tqdm progress bar.")
    p.add_argument("--input", required=True, help="Path to 3D label TIFF (Z,Y,X).")
    p.add_argument("--output", required=True, help="Path to output 3D skeleton TIFF.")
    p.add_argument("--resolution", nargs=2, type=float, default=(1.0, 1.0),
                   help="In-plane spacing (dy dx). Used by SDT.")
    p.add_argument("--alpha", type=float, default=0.8, help="Energy exponent for SDT.")
    p.add_argument("--no-relabel", action="store_true", help="Disable connected-component relabeling in SDT.")
    p.add_argument("--padding", action="store_true", help="Enable padding in SDT.")
    p.add_argument("--no-smooth", action="store_true", help="Disable edge smoothing in SDT.")
    p.add_argument("--no-smooth-skeleton-only", action="store_true",
                   help="If smoothing enabled, apply it to both semantic and skeleton in SDT.")
    p.add_argument("--bg-value", type=float, default=-1.0, help="Background value for SDT distance.")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU cores).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_slice_skeleton_parallel(
        input_tiff=args.input,
        output_tiff=args.output,
        resolution_xy=tuple(args.resolution),
        alpha=args.alpha,
        relabel=(not args.no_relabel),
        padding=args.padding,
        smooth=(not args.no_smooth),
        smooth_skeleton_only=(not args.no_smooth_skeleton_only),
        bg_value=args.bg_value,
        num_workers=args.workers,
    )
