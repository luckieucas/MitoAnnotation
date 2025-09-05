#!/usr/bin/env python3
"""
compute_detection_metrics_and_extract_patches_tiff.py

Given:
  - A CSV of matched_pairs and matched_scores (IOU)
  - A 3D image volume (tiff)
  - A 3D GT mask volume (tiff)
  - A 3D Pred mask volume (tiff)

This script will:
  1. Compute TP / FP / FN counts and F1-score based on an IOU threshold.
  2. Identify Pred FP IDs and GT FN IDs.
  3. For each of these IDs, compute the 3D bounding box, pad by N voxels on each side,
     and extract the corresponding patch from image, GT mask, and Pred mask.
  4. Save each patch trio as separate TIFF files under FP/ and FN/ subfolders.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tifffile

def parse_args():
    p = argparse.ArgumentParser(description="Compute detection metrics and extract FP/FN patches as TIFF.")
    p.add_argument("--csv",         required=True,  help="Path to CSV with columns matched_pairs, matched_scores")
    p.add_argument("--image_tiff",  required=True,  help="3D image volume (tiff)")
    p.add_argument("--gt_tiff",     required=True,  help="3D ground-truth mask (tiff)")
    p.add_argument("--pred_tiff",   required=True,  help="3D predicted mask (tiff)")
    p.add_argument("--output_dir",  required=True,  help="Directory to save extracted patches")
    p.add_argument("--iou_thresh",  type=float, default=0.5, help="IOU threshold for TP (default: 0.5)")
    p.add_argument("--padding",     type=int,   default=10,  help="Padding (voxels) around bounding box (default: 10)")
    return p.parse_args()

def load_volumes(image_path, gt_path, pred_path):
    """
    Load three 3D volumes (image, GT mask, Pred mask) from TIFF files.
    """
    img   = tifffile.imread(image_path)
    gt    = tifffile.imread(gt_path)
    pred  = tifffile.imread(pred_path)
    return img, gt, pred

def compute_metrics(df, gt_ids, pred_ids, iou_thresh=0.5):
    """
    Compute TP, FP, FN, F1 based on matched_pairs and IOU threshold.
    Also return sets of FP IDs and FN IDs.
    """
    df[["gt","pred"]] = (
        df["matched_pairs"]
        .str.strip("()")
        .str.split(",", expand=True)
        .astype(int)
    )
    df = df.rename(columns={"matched_scores":"iou"})
    # TP: IOU >= threshold
    tp_pairs = df[df["iou"] >= iou_thresh]
    TP = len(tp_pairs)
    # Low-IOU matches count as both FP and FN
    low_pairs = df[df["iou"] < iou_thresh]
    # Determine FP/FN from low-IOU matches
    # but we'll compute final sets below
    # Unmatched sets
    matched_gt   = set(df["gt"])
    matched_pred = set(df["pred"])
    FN = len(gt_ids) - TP
    FP = len(pred_ids) - TP
    F1 = 2 * TP / (2 * TP + FP + FN) if (2*TP+FP+FN) > 0 else 0.0

    # FP IDs are: those low-IoU pred IDs, plus any pred IDs not in high-IoU TP set
    fp_ids = set(low_pairs["pred"]).union(pred_ids - set(tp_pairs["pred"]))
    # FN IDs are: those low-IoU gt IDs, plus any gt IDs not in high-IoU TP set
    fn_ids = set(low_pairs["gt"]).union(gt_ids - set(tp_pairs["gt"]))

    return {"TP": TP, "FP": FP, "FN": FN, "F1": F1}, fp_ids, fn_ids

def extract_and_save_tiffs(ids, volume, ref_gt, ref_pred, output_subdir, prefix, padding):
    """
    ids: iterable of label IDs to extract
    volume: 3D image volume
    ref_gt, ref_pred: the 3D GT, Pred masks
    output_subdir: path to either FP/ or FN/ folder
    prefix: 'fp' or 'fn'
    """
    os.makedirs(output_subdir, exist_ok=True)
    Z, Y, X = volume.shape

    for lab in ids:
        # choose which mask to locate the label in
        # for FP we look into ref_pred, for FN we look into ref_gt
        mask = (ref_pred == lab) if prefix == "fp" else (ref_gt == lab)
        if not mask.any():
            continue

        zs, ys, xs = np.where(mask)
        z0, z1 = max(zs.min() - padding, 0),   min(zs.max() + padding, Z - 1)
        y0, y1 = max(ys.min() - padding, 0),   min(ys.max() + padding, Y - 1)
        x0, x1 = max(xs.min() - padding, 0),   min(xs.max() + padding, X - 1)

        # crop patches from all three volumes
        img_patch   = volume[z0:z1+1, y0:y1+1, x0:x1+1]
        gt_patch    = ref_gt[z0:z1+1, y0:y1+1, x0:x1+1]
        pred_patch  = ref_pred[z0:z1+1, y0:y1+1, x0:x1+1]

        # save each as separate TIFF
        tifffile.imwrite(os.path.join(output_subdir, f"{prefix}_{lab}_image.tiff"),  img_patch, compression="zlib")
        tifffile.imwrite(os.path.join(output_subdir, f"{prefix}_{lab}_gt.tiff"),     gt_patch, compression="zlib")
        tifffile.imwrite(os.path.join(output_subdir, f"{prefix}_{lab}_pred.tiff"),   pred_patch, compression="zlib")

        print(f"Saved {prefix.upper()} ID {lab} patches (image/gt/pred) in {output_subdir}")

def main():
    args = parse_args()

    # prepare directories
    fp_dir = os.path.join(args.output_dir, "FP")
    fn_dir = os.path.join(args.output_dir, "FN")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    # load data (image, gt, pred)
    img, gt, pred = load_volumes(
        args.image_tiff,
        args.gt_tiff,
        args.pred_tiff
    )

    # compute unique IDs (exclude background 0)
    gt_ids   = set(np.unique(gt))   - {0}
    pred_ids = set(np.unique(pred)) - {0}

    # read matched CSV and compute metrics
    df = pd.read_csv(args.csv)
    metrics, fp_ids, fn_ids = compute_metrics(df, gt_ids, pred_ids, args.iou_thresh)

    print("=== Detection Metrics ===")
    print(f"TP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}, F1 = {metrics['F1']:.4f}")

    # extract & save patches for FP and FN
    extract_and_save_tiffs(fp_ids, img, gt, pred, fp_dir, prefix="fp", padding=args.padding)
    extract_and_save_tiffs(fn_ids, img, gt, pred, fn_dir, prefix="fn", padding=args.padding)

if __name__ == "__main__":
    main()
