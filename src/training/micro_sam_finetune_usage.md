# Micro SAM Fine-tuning Script Usage

## Overview
This script fine-tunes Segment Anything Model (SAM) using nnUNet formatted datasets.

## Features
- Automatically checks for and creates 2D slices from 3D TIFF data
- Auto-generates checkpoint names from dataset directory
- Exports trained model to checkpoint directory
- Configurable training parameters via command-line arguments

## Basic Usage

### Minimal Example (Required Arguments Only)
```bash
python micro_sam_finetune.py \
    --dataset_path /path/to/Dataset005_MitoHardKidney
```

This will:
- Use default model type: `vit_b`
- Auto-generate checkpoint name: `sam_vit_b_Dataset005_MitoHardKidney`
- Save to: `./checkpoints/sam_vit_b_Dataset005_MitoHardKidney/`
- Export model to: `./checkpoints/sam_vit_b_Dataset005_MitoHardKidney/sam_vit_b_Dataset005_MitoHardKidney_exported.pth`

### With Custom Model Type
```bash
python micro_sam_finetune.py \
    --dataset_path /path/to/Dataset005_MitoHardKidney \
    --model_type vit_h
```

This will use `vit_h` model and checkpoint name will be: `sam_vit_h_Dataset005_MitoHardKidney`

### Full Custom Configuration
```bash
python micro_sam_finetune.py \
    --dataset_path /path/to/Dataset005_MitoHardKidney \
    --model_type vit_b \
    --checkpoint_name my_custom_sam_model \
    --n_epochs 30 \
    --batch_size 2 \
    --patch_size 1024 \
    --n_objects_per_batch 50
```

## Arguments

### Required
- `--dataset_path`: Path to the nnUNet dataset root directory (must contain imagesTr, instancesTr, imagesTs, instancesTs)

### Optional
- `--model_type`: SAM model type (default: `vit_b`)
  - Choices: `vit_b`, `vit_l`, `vit_h`, `vit_t`, `vit_b_lm`, `vit_l_lm`, `vit_h_lm`
  - Note: `vit_h` yields higher quality but trains slower
  
- `--checkpoint_name`: Custom checkpoint name (default: auto-generated from dataset name)

- `--n_epochs`: Number of training epochs (default: 20)

- `--batch_size`: Training batch size (default: 1)

- `--patch_size`: Patch size for training (default: 512)

- `--n_objects_per_batch`: Number of objects per batch to be sampled (default: 25)

- `--no_instance_segmentation`: Disable training additional convolutional decoder

- `--skip_export`: Skip exporting the model after training

## Dataset Structure

Your dataset should follow the nnUNet format:
```
Dataset005_MitoHardKidney/
├── imagesTr/          # Training images (3D TIFF)
├── instancesTr/       # Training instance labels (3D TIFF)
├── imagesTs/          # Test images (3D TIFF, used as validation)
└── instancesTs/       # Test instance labels (3D TIFF, used as validation)
```

The script will automatically create:
```
Dataset005_MitoHardKidney/
└── 2d_slices/
    ├── train/
    │   └── {volume_name}/
    │       ├── images/
    │       └── masks/
    └── val/
        └── {volume_name}/
            ├── images/
            └── masks/
```

## Examples

### Example 1: Quick Start with Default Settings
```bash
python micro_sam_finetune.py \
    --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset005_MitoHardKidney
```

### Example 2: High Quality Training (using vit_h)
```bash
python micro_sam_finetune.py \
    --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset005_MitoHardKidney \
    --model_type vit_h \
    --n_epochs 50 \
    --patch_size 1024
```

### Example 3: Fast Prototyping (fewer epochs)
```bash
python micro_sam_finetune.py \
    --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset005_MitoHardKidney \
    --model_type vit_b \
    --n_epochs 10 \
    --checkpoint_name sam_quick_test
```

### Example 4: Training Without Instance Segmentation Decoder
```bash
python micro_sam_finetune.py \
    --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset005_MitoHardKidney \
    --no_instance_segmentation
```

## Output

After training, you will find:
1. Checkpoint directory: `./checkpoints/{checkpoint_name}/`
   - `best.pt`: Best model checkpoint
   - `{checkpoint_name}_exported.pth`: Exported model for inference
   - Training logs and other checkpoints

## Tips

1. **GPU Memory**: If you encounter OOM errors, reduce `--batch_size` or `--patch_size`
2. **Training Speed**: Use `vit_b` for faster training, `vit_h` for better quality
3. **Reusing Data**: The script checks for existing `2d_slices` folder and reuses it if available
4. **Custom Names**: Use `--checkpoint_name` to organize multiple experiments

## Features

### Automatic Empty Patch Filtering
The script automatically:
- Filters out 2D slices that are entirely background (no annotations)
- Rejects training patches that have less than 1% foreground pixels
- This prevents "No foreground objects were found" errors during training
- Ensures efficient training by focusing on informative regions

