# FP/FN 3D Analysis Tool

This tool performs False Positive (FP) and False Negative (FN) analysis on predicted masks and generates a comprehensive PDF with 3D visualizations.

## Features

- **FP/FN Detection**: Automatically identifies FP and FN cases based on IoU threshold
- **3D Visualization**: Renders GT and Pred masks in 3D from multiple viewpoints
- **PDF Generation**: Creates a multi-page PDF with each FP/FN case on a separate page
- **Flexible Rendering**: Supports both VTK-based and matplotlib-based 3D rendering
- **Comprehensive Metrics**: Computes TP, FP, FN, and F1 scores

## Requirements

### Required Dependencies
- numpy
- pandas
- tifffile
- PIL (Pillow)
- matplotlib
- scikit-image (for marching_cubes)

### Optional Dependencies
- VTK (for high-quality 3D rendering)
- vtkmodules

## Usage

### Basic Usage

```bash
python fp_fn_3d_analysis.py \
    --csv_file matches.csv \
    --gt_mask gt_mask.tiff \
    --pred_mask pred_mask.tiff \
    --img image.tiff \
    --output_pdf fp_fn_analysis.pdf
```

### Advanced Usage

```bash
python fp_fn_3d_analysis.py \
    --csv_file matches.csv \
    --gt_mask gt_mask.tiff \
    --pred_mask pred_mask.tiff \
    --img image.tiff \
    --output_pdf fp_fn_analysis.pdf \
    --iou_thresh 0.5 \
    --margins 10 20 20 \
    --smooth_iters 15 \
    --render_size 400 \
    --use_matplotlib
```

## Parameters

- `--csv_file`: CSV file containing matched_pairs and matched_scores columns
- `--gt_mask`: 3D ground-truth mask (TIFF format)
- `--pred_mask`: 3D predicted mask (TIFF format)  
- `--img`: 3D image volume (TIFF format)
- `--output_pdf`: Output PDF file path
- `--iou_thresh`: IoU threshold for True Positive classification (default: 0.5)
- `--margins`: Padding margins [z, y, x] around bounding boxes (default: [10, 20, 20])
- `--smooth_iters`: Number of smoothing iterations for 3D rendering (default: 15)
- `--render_size`: Size of 3D render images (default: 400)
- `--use_matplotlib`: Use matplotlib instead of VTK for 3D rendering

## Output

The tool generates:

1. **Console Output**: Detection metrics (TP, FP, FN, F1) and processing progress
2. **PDF File**: Multi-page PDF with each FP/FN case showing:
   - Case type (FP or FN) and ID
   - GT and Pred 3D renderings from 4 viewpoints (front, side, top, iso)
   - Color-coded labels for easy comparison

## CSV Format

The input CSV should contain columns:
- `matched_pairs`: String representation of (gt_id, pred_id) tuples
- `matched_scores`: IoU scores between matched pairs

Example:
```csv
matched_pairs,matched_scores
"(1, 5)",0.85
"(2, 7)",0.42
"(3, 9)",0.78
```

## FP/FN Logic

- **FP (False Positive)**: Predicted objects with low IoU or no GT match
- **FN (False Negative)**: GT objects with low IoU or no prediction match
- **TP (True Positive)**: Predicted objects with IoU >= threshold

## Rendering Options

### VTK Rendering (Default)
- High-quality 3D surface rendering
- Smooth surfaces with configurable smoothing
- Requires VTK installation

### Matplotlib Rendering (Fallback)
- Uses marching cubes for surface extraction
- Good for basic visualization
- Works without VTK

## Troubleshooting

1. **VTK Import Error**: Use `--use_matplotlib` flag
2. **Memory Issues**: Reduce `--render_size` parameter
3. **No 3D Rendering**: Ensure scikit-image is installed for marching_cubes

## Example Output Structure

```
=== Detection Metrics ===
TP = 45, FP = 12, FN = 8, F1 = 0.8154
Found 12 FP cases and 8 FN cases
Processing FP case 23...
Processing FP case 45...
...
PDF saved → fp_fn_analysis.pdf
```
