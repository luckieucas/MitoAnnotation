#!/usr/bin/env python3
"""
This script reads a CSV file of matched_pairs and matched_scores,
filters rows below a threshold, then for each gt ID:
  1) crops GT, Pred masks and the image volume by an expanded bbox
  2) in GT crop, retains only the GT ID
   3) saves each crop as 3D TIFF
  4) generates a 1×2 panel PDF of 3D surface renderings from a fixed viewpoint,
     with labels color-coded for multiple instances. Each page shows one ID pair.
"""
import argparse
import ast
import os
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage.measure import marching_cubes


def load_matches(csv_path):
    df = pd.read_csv(csv_path)
    pairs = df['matched_pairs'].apply(ast.literal_eval).tolist()
    scores = df['matched_scores'].astype(float).tolist()
    return pairs, scores


def filter_ids(pairs, scores, thresh):
    filtered = [pair for pair, score in zip(pairs, scores) if score < thresh]
    gt_ids = [gt for gt, _ in filtered]
    pred_ids = [pr for _, pr in filtered]
    return gt_ids, pred_ids


def compute_bbox(mask, label_id):
    coords = np.where(mask == label_id)
    if coords[0].size == 0:
        raise ValueError(f"Label ID {label_id} not found in mask")
    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()
    return (z0, z1), (y0, y1), (x0, x1)


def expand_bbox(bbox, margins, shape):
    (z0, z1), (y0, y1), (x0, x1) = bbox
    dz, dy, dx = margins
    depth, height, width = shape
    z0e, z1e = max(0,z0-dz), min(depth-1, z1+dz)
    y0e, y1e = max(0,y0-dy), min(height-1, y1+dy)
    x0e, x1e = max(0,x0-dx), min(width-1, x1+dx)
    return (z0e,z1e), (y0e,y1e), (x0e,x1e)


def crop_region(volume, bbox_exp):
    (z0, z1),(y0, y1),(x0, x1) = bbox_exp
    return volume[z0:z1+1, y0:y1+1, x0:x1+1]


def save_3d_visual(gt_crop, pred_crop, gt_id, pred_id, pdf_pages):
    fig = plt.figure(figsize=(10,5))
    cmap = plt.get_cmap('tab20')
    titles = ['GT', 'Pred']
    crops = [gt_crop, pred_crop]
    
    # Add main title for the page
    fig.suptitle(f'GT ID: {gt_id}, Pred ID: {pred_id}', fontsize=16, fontweight='bold')
    
    for i, crop in enumerate(crops):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        labels = np.unique(crop)
        for label in labels:
            if label == 0: continue
            verts, faces, normals, _ = marching_cubes(crop==label, level=0)
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], faces, verts[:, 2],
                color=cmap(label % 20), lw=0, alpha=0.7
            )
        ax.view_init(elev=30, azim=45)
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    
    # Save to PDF page
    pdf_pages.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crop masks/images and render 3D views.'
    )
    parser.add_argument('--csv_file',   required=True)
    parser.add_argument('--gt_mask',    required=True)
    parser.add_argument('--pred_mask',  required=True)
    parser.add_argument('--img',        required=True)
    parser.add_argument('--thresh',     type=float, default=0.5)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_pdf', required=True, help='Output PDF file path')
    parser.add_argument('--margins',    nargs=3, type=int, default=[10,20,20])
    args = parser.parse_args()

    pairs, scores = load_matches(args.csv_file)
    gt_ids, pred_ids = filter_ids(pairs, scores, args.thresh)

    gt_vol     = tiff.imread(args.gt_mask)
    pred_vol   = tiff.imread(args.pred_mask)
    img_vol    = tiff.imread(args.img)
    shape = gt_vol.shape
    os.makedirs(args.output_dir, exist_ok=True)

    # Create PDF file
    with PdfPages(args.output_pdf) as pdf_pages:
        for gt_id, pred_id in zip(gt_ids, pred_ids):
            bbox    = compute_bbox(gt_vol, gt_id)
            bb_exp  = expand_bbox(bbox, args.margins, shape)

            raw_gt     = crop_region(gt_vol, bb_exp)
            raw_pred   = crop_region(pred_vol, bb_exp)
            raw_img    = crop_region(img_vol, bb_exp)

            gt_crop = np.where(raw_gt==gt_id, raw_gt, 0)
            # Save volumes
            base = f'gt_{gt_id:04d}'
            tiff.imwrite(os.path.join(args.output_dir, f'{base}_gt.tiff'), gt_crop)
            tiff.imwrite(os.path.join(args.output_dir, f'{base}_pred.tiff'), raw_pred)
            tiff.imwrite(os.path.join(args.output_dir, f'{base}_img.tiff'), raw_img)

            # Save 3D render to PDF page
            save_3d_visual(gt_crop, raw_pred, gt_id, pred_id, pdf_pages)
            print(f'Added page for GT ID {gt_id}, Pred ID {pred_id} to PDF')

    print(f'PDF saved → {args.output_pdf}')
