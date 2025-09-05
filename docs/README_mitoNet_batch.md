# MitoNet Batch Processing for nnUNet Raw Directory

This script has been modified to process multiple datasets in nnUNet raw directory format.

## Directory Structure Expected

```
nnUNet_raw/
├── Dataset001/
│   ├── imagesTs/
│   │   ├── image1.tif
│   │   ├── image2.tif
│   │   └── ...
│   └── instancesTs/
│       ├── image1.tif
│       ├── image2.tif
│       └── ...
├── Dataset002/
│   ├── imagesTs/
│   └── instancesTs/
└── ...
```

## Usage

### Basic Usage

```bash
python mitoNet_baseline.py /path/to/nnUNet_raw --datasets Dataset001 Dataset002 --config_path configs/MitoNet_v1.yaml
```

### With GPU and Evaluation

```bash
python mitoNet_baseline.py /path/to/nnUNet_raw \
    --datasets Dataset001 Dataset002 Dataset003 \
    --config_path configs/MitoNet_v1.yaml \
    --use_gpu \
    --evaluate \
    --confidence_thr 0.5 \
    --center_confidence_thr 0.1 \
    --min_size 500 \
    --min_extent 5
```

### Skip Existing Predictions (Default Behavior)

By default, the script will skip files that already have predictions and only run evaluation:

```bash
python mitoNet_baseline.py /path/to/nnUNet_raw \
    --datasets Dataset001 Dataset002 \
    --evaluate
```

### Force Re-prediction

To force re-prediction even if prediction files exist:

```bash
python mitoNet_baseline.py /path/to/nnUNet_raw \
    --datasets Dataset001 Dataset002 \
    --force_repredict \
    --evaluate
```

### Using the Example Script

1. Edit `run_mitoNet_batch.py` to set your paths and dataset names
2. Run: `python run_mitoNet_batch.py`

## Parameters

- `input_path`: Path to the nnUNet raw directory containing datasets
- `--datasets`: List of dataset names to process (e.g., Dataset001 Dataset002)
- `--config_path`: Path to the model config file (default: configs/MitoNet_v1.yaml)
- `--use_gpu`: Use GPU for inference if available
- `--evaluate`: Run evaluation after prediction (requires instancesTs folder)
- `--force_repredict`: Force re-prediction even if prediction files exist
- `--confidence_thr`: Segmentation confidence threshold (default: 0.5)
- `--center_confidence_thr`: Center confidence threshold (default: 0.1)
- `--min_distance_object_centers`: Minimum distance between object centers (default: 3)
- `--min_size`: Minimum size of objects in voxels (default: 500)
- `--min_extent`: Minimum bounding box extent for objects (default: 5)
- `--pixel_vote_thr`: Voxel vote threshold for ortho-plane consensus (default: 2)

## Output Structure

For each processed dataset, the script creates:

```
Dataset001/
├── imagesTs/           # Original input images
├── instancesTs/        # Ground truth (if available)
└── imagesTs_mitoNet_pred/  # Generated predictions
    ├── image1_xy.tif
    ├── image1_xz.tif
    ├── image1_yz.tif
    ├── image1_consensus.tif
    ├── image1_consensus.txt  # Evaluation metrics (if --evaluate)
    ├── image1_consensus.csv  # Detailed evaluation results (if --evaluate)
    └── ...
```

## Evaluation

When `--evaluate` flag is used, the script will:

1. Look for corresponding ground truth files in `instancesTs/` folder
2. Run evaluation using the `evaluate_res` module
3. Save evaluation metrics as `.txt` and `.csv` files
4. Print F1 score, precision, and recall for each processed image

## Requirements

- empanada
- empanada_napari
- tifffile
- torch
- numpy
- tqdm
- connectomics (for evaluation)
- scikit-image (for evaluation)
- pandas (for evaluation)

## Smart Skip Feature

The script includes intelligent skipping functionality:

- **Default behavior**: If prediction files already exist, the script skips prediction and only runs evaluation
- **Force re-prediction**: Use `--force_repredict` to overwrite existing predictions
- **Statistics**: The script provides detailed statistics for each dataset showing:
  - Total files found
  - Files skipped (already exist)
  - Files processed
  - Errors encountered

## Notes

- The script processes all `.tif` and `.tiff` files in the `imagesTs` folder
- Only 3D images are processed (2D images are skipped with a warning)
- If ground truth files are not found, evaluation is skipped for that image
- The script continues processing even if individual images fail
- Existing predictions are automatically detected and skipped unless `--force_repredict` is used
