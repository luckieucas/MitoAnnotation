#!/usr/bin/env python3
"""
FP/FN 3D Analysis Script

This script performs FP/FN analysis on predicted masks and generates a PDF with 3D visualizations.
It combines:
1. CSV matching analysis from fp_fn_analysis.py
2. 3D rendering capabilities from mito_mask_visualize_3d.py and mask_visulize_3d.py
3. PDF generation from error_analysis.py

For each FP/FN case, it:
1. Extracts the 3D bounding box with padding
2. Renders GT and Pred masks in 3D from multiple viewpoints
3. Saves each case as a PDF page with GT vs Pred comparison
"""

import os
import sys
import argparse
import ast
import numpy as np
import pandas as pd
import tifffile as tiff
import colorsys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List, Tuple, Dict, Optional
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# VTK imports
try:
    import vtk
    from vtkmodules.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    print("[WARNING] VTK not available. 3D rendering will be disabled.")
    VTK_AVAILABLE = False


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors using HSV color space."""
    if n == 0:
        return []
    
    hsv_colors = []
    golden_ratio_conjugate = 0.61803398875
    hue = 0.5
    for _ in range(n):
        hue += golden_ratio_conjugate
        hue %= 1
        hsv_colors.append((hue, 0.85, 0.95))

    rgb_colors = []
    for h, s, v in hsv_colors:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb_colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
    return rgb_colors


def relabel_data(data: np.ndarray):
    """Relabel all non-zero labels to 1,2,3... and return new data"""
    unique = np.unique(data)
    unique = unique[unique != 0]
    mapping = {old: new for new, old in enumerate(unique, start=1)}
    out = np.zeros_like(data, dtype=np.uint16)
    for old, new in mapping.items():
        out[data == old] = new
    return out


def load_matches(csv_path):
    """Load matched pairs and scores from CSV file."""
    df = pd.read_csv(csv_path)
    pairs = df['matched_pairs'].apply(ast.literal_eval).tolist()
    scores = df['matched_scores'].astype(float).tolist()
    return pairs, scores


def compute_bbox(mask, label_id):
    """Compute bounding box for a specific label ID."""
    coords = np.where(mask == label_id)
    if coords[0].size == 0:
        raise ValueError(f"Label ID {label_id} not found in mask")
    z0, z1 = coords[0].min(), coords[0].max()
    y0, y1 = coords[1].min(), coords[1].max()
    x0, x1 = coords[2].min(), coords[2].max()
    return (z0, z1), (y0, y1), (x0, x1)


def expand_bbox(bbox, margins, shape):
    """Expand bounding box by margins, respecting volume boundaries."""
    (z0, z1), (y0, y1), (x0, x1) = bbox
    dz, dy, dx = margins
    depth, height, width = shape
    z0e, z1e = max(0, z0-dz), min(depth-1, z1+dz)
    y0e, y1e = max(0, y0-dy), min(height-1, y1+dy)
    x0e, x1e = max(0, x0-dx), min(width-1, x1+dx)
    return (z0e, z1e), (y0e, y1e), (x0e, x1e)


def crop_region(volume, bbox_exp):
    """Crop region from volume using expanded bounding box."""
    (z0, z1), (y0, y1), (x0, x1) = bbox_exp
    return volume[z0:z1+1, y0:y1+1, x0:x1+1]


def compute_fp_fn_metrics(pairs, scores, gt_ids, pred_ids, iou_thresh=0.5):
    """Compute FP/FN metrics and return FP and FN ID sets."""
    # Convert pairs to DataFrame for easier processing
    df_data = []
    for (gt_id, pred_id), score in zip(pairs, scores):
        df_data.append({'gt': gt_id, 'pred': pred_id, 'iou': score})
    df = pd.DataFrame(df_data)
    
    # TP: IOU >= threshold
    tp_pairs = df[df["iou"] >= iou_thresh]
    TP = len(tp_pairs)
    
    # Low-IOU matches count as both FP and FN
    low_pairs = df[df["iou"] < iou_thresh]
    
    # Unmatched sets
    matched_gt = set(df["gt"])
    matched_pred = set(df["pred"])
    
    # FP IDs: low-IoU pred IDs + pred IDs not in high-IoU TP set
    fp_ids = set(low_pairs["pred"]).union(pred_ids - set(tp_pairs["pred"]))
    # FN IDs: low-IoU gt IDs + gt IDs not in high-IoU TP set  
    fn_ids = set(low_pairs["gt"]).union(gt_ids - set(tp_pairs["gt"]))
    
    FN = len(fn_ids)
    FP = len(fp_ids)
    F1 = 2 * TP / (2 * TP + FP + FN) if (2*TP+FP+FN) > 0 else 0.0
    
    return {"TP": TP, "FP": FP, "FN": FN, "F1": F1}, fp_ids, fn_ids


def numpy_to_vtk_image(data: np.ndarray) -> vtk.vtkImageData:
    """Convert 3D numpy array to vtkImageData."""
    data = np.ascontiguousarray(data)
    depth, height, width = data.shape
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(width, height, depth)
    vtk_array = numpy_support.numpy_to_vtk(
        num_array=data.ravel(order='C'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_SHORT if data.dtype == np.uint16 else vtk.VTK_UNSIGNED_CHAR
    )
    vtk_img.GetPointData().SetScalars(vtk_array)
    return vtk_img


def process_label(label: int, data: np.ndarray, smooth_iters: int, color_map: Dict[int, Tuple[int, int, int]]) -> vtk.vtkActor:
    """Extract isosurface for a single label and return corresponding Actor."""
    if label == 0:
        return None
    mask = np.where(data == label, label, 0).astype(data.dtype)
    vtk_img = numpy_to_vtk_image(mask)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, label)
    mc.ComputeNormalsOn()
    mc.Update()

    poly_data = mc.GetOutput()
    if smooth_iters > 0 and poly_data.GetNumberOfPoints() > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(mc.GetOutputPort())
        smoother.SetNumberOfIterations(smooth_iters)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()
        poly_data = smoother.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    r, g, b = color_map.get(label, (255, 255, 255))
    actor.GetProperty().SetColor(r/255.0, g/255.0, b/255.0)
    actor.GetProperty().SetOpacity(1.0)
    return actor


def render_single_view(data: np.ndarray, smooth_iters: int, color_map: Dict,
                       az: float, el: float, size=(800, 800)) -> Image.Image:
    """Render a single 3D view from specified viewpoint (azimuth, elevation)."""
    if not VTK_AVAILABLE:
        return Image.new('RGB', size, (255, 255, 255))
        
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renwin = vtk.vtkRenderWindow()
    renwin.OffScreenRenderingOn()
    renwin.AddRenderer(renderer)
    renwin.SetSize(*size)

    labels = [l for l in np.unique(data) if l != 0]
    if not labels:
        return Image.new('RGB', size, (255, 255, 255))
    
    with ThreadPoolExecutor() as exe:
        futures = [exe.submit(process_label, lab, data, smooth_iters, color_map) for lab in labels]
        for f in futures:
            actor = f.result()
            if actor:
                renderer.AddActor(actor)

    if renderer.GetActors().GetNumberOfItems() == 0:
        return Image.new('RGB', size, (255, 255, 255))

    cam = renderer.GetActiveCamera()
    renderer.ResetCamera()
    cam.Azimuth(az)
    cam.Elevation(el)
    renderer.ResetCameraClippingRange()
    renwin.Render()

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    vtk_img = w2if.GetOutput()

    width, height, _ = vtk_img.GetDimensions()
    arr = numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    arr = arr.reshape(height, width, 3)
    return Image.fromarray(np.flipud(arr))


def get_font(font_size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """Get an available font."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf", 
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try: 
                return ImageFont.truetype(path, font_size)
            except IOError: 
                continue
    return ImageFont.load_default()


def annotate_image(img: Image.Image, label: str, font_size: int = 30) -> Image.Image:
    """Add title to image."""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)
        
    img_w, _ = img.size
    position = ((img_w - text_w) // 2, 10)
    draw.rectangle((position[0] - 5, position[1] - 5, position[0] + text_w + 5, position[1] + text_h + 5), fill='white')
    draw.text(position, label, fill='black', font=font)
    return img


def create_fp_fn_page(gt_crop, pred_crop, case_type, case_id, smooth_iters, render_size):
    """Create a single page showing GT vs Pred comparison for FP/FN case."""
    # Generate colors for labels
    gt_labels = [l for l in np.unique(gt_crop) if l != 0]
    pred_labels = [l for l in np.unique(pred_crop) if l != 0]
    all_labels = list(set(gt_labels + pred_labels))
    
    if not all_labels:
        return None
        
    colors = generate_distinct_colors(len(all_labels))
    color_map = {label: color for label, color in zip(all_labels, colors)}
    
    # Define viewpoints
    views = {
        'front': (0, 0),
        'side': (90, 0), 
        'top': (0, 90),
        'iso': (45, 45)
    }
    
    # Render GT and Pred for each view
    images = []
    titles = []
    
    for view_name, (az, el) in views.items():
        # GT view
        gt_img = render_single_view(gt_crop, smooth_iters, color_map, az, el, (render_size, render_size))
        gt_img = annotate_image(gt_img, f"GT ({view_name})", font_size=24)
        
        # Pred view  
        pred_img = render_single_view(pred_crop, smooth_iters, color_map, az, el, (render_size, render_size))
        pred_img = annotate_image(pred_img, f"Pred ({view_name})", font_size=24)
        
        images.extend([gt_img, pred_img])
        titles.extend([f"GT {view_name}", f"Pred {view_name}"])
    
    # Create 2x4 grid layout
    w, h = render_size, render_size
    page_img = Image.new('RGB', (w * 2, h * 4 + 100), (255, 255, 255))
    
    # Add main title
    draw = ImageDraw.Draw(page_img)
    main_font = get_font(36)
    main_title = f"{case_type.upper()} Case ID: {case_id}"
    try:
        bbox = draw.textbbox((0, 0), main_title, font=main_font)
        title_w = bbox[2] - bbox[0]
    except AttributeError:
        title_w, _ = draw.textsize(main_title, font=main_font)
    
    draw.text(((w * 2 - title_w) // 2, 20), main_title, fill='black', font=main_font)
    
    # Paste images in 2x4 grid
    for i, img in enumerate(images):
        row = i // 2
        col = i % 2
        page_img.paste(img, (col * w, 100 + row * h))
    
    return page_img


def save_3d_visual_matplotlib(gt_crop, pred_crop, case_type, case_id, pdf_pages):
    """Alternative 3D visualization using matplotlib (fallback when VTK unavailable)."""
    try:
        from skimage.measure import marching_cubes
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("[WARNING] skimage not available for marching_cubes. Using simple visualization.")
        return
    
    fig = plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap('tab20')
    titles = ['GT', 'Pred']
    crops = [gt_crop, pred_crop]
    
    # Add main title for the page
    fig.suptitle(f'{case_type.upper()} Case ID: {case_id}', fontsize=16, fontweight='bold')
    
    for i, crop in enumerate(crops):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        labels = np.unique(crop)
        for label in labels:
            if label == 0: 
                continue
            try:
                verts, faces, normals, _ = marching_cubes(crop==label, level=0)
                ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    color=cmap(label % 20), lw=0, alpha=0.7
                )
            except:
                # Skip if marching cubes fails
                continue
        ax.view_init(elev=30, azim=45)
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    pdf_pages.savefig(fig, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='FP/FN 3D Analysis: Generate PDF with 3D visualizations of FP and FN cases.'
    )
    parser.add_argument('--csv_file', required=True, help='CSV file with matched_pairs and matched_scores')
    parser.add_argument('--gt_mask', required=True, help='3D ground-truth mask (tiff)')
    parser.add_argument('--pred_mask', required=True, help='3D predicted mask (tiff)')
    parser.add_argument('--img', required=True, help='3D image volume (tiff)')
    parser.add_argument('--output_pdf', required=True, help='Output PDF file path')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU threshold for TP (default: 0.5)')
    parser.add_argument('--margins', nargs=3, type=int, default=[10, 20, 20], help='Padding margins [z, y, x]')
    parser.add_argument('--smooth_iters', type=int, default=15, help='3D smoothing iterations')
    parser.add_argument('--render_size', type=int, default=400, help='3D render size')
    parser.add_argument('--use_matplotlib', action='store_true', help='Use matplotlib instead of VTK for 3D rendering')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    pairs, scores = load_matches(args.csv_file)
    gt_vol = tiff.imread(args.gt_mask)
    pred_vol = tiff.imread(args.pred_mask)
    img_vol = tiff.imread(args.img)
    shape = gt_vol.shape
    
    # Get unique IDs
    gt_ids = set(np.unique(gt_vol)) - {0}
    pred_ids = set(np.unique(pred_vol)) - {0}
    
    # Compute FP/FN metrics
    print("Computing FP/FN metrics...")
    metrics, fp_ids, fn_ids = compute_fp_fn_metrics(pairs, scores, gt_ids, pred_ids, args.iou_thresh)
    
    print("=== Detection Metrics ===")
    print(f"TP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}, F1 = {metrics['F1']:.4f}")
    print(f"Found {len(fp_ids)} FP cases and {len(fn_ids)} FN cases")
    
    # Create PDF
    print(f"Generating PDF: {args.output_pdf}")
    with PdfPages(args.output_pdf) as pdf_pages:
        # Process FP cases
        for fp_id in sorted(fp_ids):
            print(f"Processing FP case {fp_id}...")
            try:
                # For FP, we look at the pred mask to find the bounding box
                bbox = compute_bbox(pred_vol, fp_id)
                bb_exp = expand_bbox(bbox, args.margins, shape)
                
                # Crop regions
                raw_gt = crop_region(gt_vol, bb_exp)
                raw_pred = crop_region(pred_vol, bb_exp)
                raw_img = crop_region(img_vol, bb_exp)
                
                # Filter GT to only show overlapping regions
                gt_crop = np.where(raw_pred == fp_id, raw_gt, 0)
                pred_crop = np.where(raw_pred == fp_id, raw_pred, 0)
                
                if args.use_matplotlib or not VTK_AVAILABLE:
                    save_3d_visual_matplotlib(gt_crop, pred_crop, 'FP', fp_id, pdf_pages)
                else:
                    page_img = create_fp_fn_page(gt_crop, pred_crop, 'FP', fp_id, args.smooth_iters, args.render_size)
                    if page_img:
                        pdf_pages.savefig(plt.figure(figsize=(page_img.width/100, page_img.height/100)))
                        plt.close()
                
            except Exception as e:
                print(f"Error processing FP case {fp_id}: {e}")
                continue
        
        # Process FN cases
        for fn_id in sorted(fn_ids):
            print(f"Processing FN case {fn_id}...")
            try:
                # For FN, we look at the gt mask to find the bounding box
                bbox = compute_bbox(gt_vol, fn_id)
                bb_exp = expand_bbox(bbox, args.margins, shape)
                
                # Crop regions
                raw_gt = crop_region(gt_vol, bb_exp)
                raw_pred = crop_region(pred_vol, bb_exp)
                raw_img = crop_region(img_vol, bb_exp)
                
                # Filter Pred to only show overlapping regions
                gt_crop = np.where(raw_gt == fn_id, raw_gt, 0)
                pred_crop = np.where(raw_gt == fn_id, raw_pred, 0)
                
                if args.use_matplotlib or not VTK_AVAILABLE:
                    save_3d_visual_matplotlib(gt_crop, pred_crop, 'FN', fn_id, pdf_pages)
                else:
                    page_img = create_fp_fn_page(gt_crop, pred_crop, 'FN', fn_id, args.smooth_iters, args.render_size)
                    if page_img:
                        pdf_pages.savefig(plt.figure(figsize=(page_img.width/100, page_img.height/100)))
                        plt.close()
                
            except Exception as e:
                print(f"Error processing FN case {fn_id}: {e}")
                continue
    
    print(f"PDF saved → {args.output_pdf}")


if __name__ == '__main__':
    main()
