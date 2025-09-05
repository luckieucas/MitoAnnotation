# MitoAnnotation - Mitochondria Segmentation and Annotation Toolkit

This is a comprehensive toolkit for mitochondria segmentation, annotation, and analysis, including data processing, model training, prediction, post-processing, and evaluation functionalities.

## Project Structure

```
MitoAnnotation/
├── src/                          # Source code directory
│   ├── data/                     # Data processing related
│   │   ├── convert_h5_to_tiff.py
│   │   ├── convert_tiff_to_h5.py
│   │   ├── zarr_to_tiff_converter.py
│   │   └── resize_img.py
│   │
│   ├── models/                   # Model related
│   │   ├── mitoNet_baseline.py
│   │   ├── mitoNet_finetune.py
│   │   └── run_mitoNet_batch.py
│   │
│   ├── training/                 # Training related
│   │   ├── auto_train_nnunet.py
│   │   ├── run_training_direct.py
│   │   └── simple_training_example.py
│   │
│   ├── postprocessing/           # Post-processing
│   │   ├── bc_watershed.py
│   │   ├── generate_contour.py
│   │   ├── postprocessing.py
│   │   ├── refine_instances.py
│   │   ├── predict_boundary.py
│   │   └── mask_cc3d.py
│   │
│   ├── evaluation/               # Evaluation and analysis
│   │   ├── evaluate_res.py
│   │   ├── evaluate_dataset_difficult.py
│   │   ├── error_analysis.py
│   │   ├── fp_fn_analysis.py
│   │   └── fp_fn_3d_analysis.py
│   │
│   ├── visualization/            # Visualization
│   │   ├── mask_visulize_3d.py
│   │   ├── mito_mask_visualize_3d.py
│   │   └── generate_complexity_figure.py
│   │
│   └── utils/                    # Utility functions
│       ├── filter_pred_by_mask.py
│       └── filter_mito_by_mask_id.py
│
├── configs/                      # Configuration files
│   ├── MitoNet_v1.yaml
│   ├── mitohard.yml
│   ├── mitohard_compatible.yml
│   ├── mitohard_final.yml
│   ├── mitohard_simple.yml
│   ├── sam.yml
│   ├── sam2_mito.yml
│   └── empanada_env.yml
│
├── scripts/                      # Script files
│   ├── README_SLURM.md
│   ├── submit_auto_training.sl
│   ├── submit_evaluate_res.sl
│   ├── submit_mitonet_zs-baseline.sl
│   ├── submit_nnunet_baseline.sl
│   ├── submit_step_by_step.sl
│   ├── train_nnunet.sl
│   ├── compute_mito_complexity_metrics.sl
│   └── submit_jobs.sh
│
├── docs/                         # Detailed documentation
│   ├── README_AUTO_TRAIN.md
│   ├── README_FILTER_MITO.md
│   ├── README_fp_fn_3d_analysis.md
│   ├── README_mitohard.md
│   └── README_mitoNet_batch.md
│
├── examples/                     # Example scripts
│   └── run_training_example.sh
│
├── checkpoints/                  # Model checkpoints
├── logs/                         # Log files
├── figs/                         # Image outputs
├── requirements.txt
└── README.md
```

## Quick Start

### Environment Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create Conda Environment** (Recommended):
```bash
# Use the merged environment configuration
conda env create -f configs/mitohard_compatible.yml
conda activate mitohard
```

### Basic Usage

#### 1. Data Preprocessing

**Resize Images**:
```bash
python src/data/resize_img.py --input <input_path> --output <output_path> --size 443 606 870 --mode mask
```

**Convert Data Formats**:
```bash
# H5 to TIFF
python src/data/convert_h5_to_tiff.py

# TIFF to H5
python src/data/convert_tiff_to_h5.py
```

#### 2. Model Training

**Auto-train nnUNet**:
```bash
python src/training/auto_train_nnunet.py \
    --data_folder "/path/to/data" \
    --output_dir "/path/to/output" \
    --dataset_id "Dataset001_MitoLE"
```

**Use MitoNet**:
```bash
python src/models/mitoNet_baseline.py /path/to/nnUNet_raw \
    --datasets Dataset001 Dataset002 \
    --config_path configs/MitoNet_v1.yaml \
    --use_gpu \
    --evaluate
```

#### 3. Post-processing

**Generate Boundaries**:
```bash
python src/postprocessing/generate_contour.py -i <input_dir> -o <output_dir> -w 5
```

**BC Watershed**:
```bash
python src/postprocessing/bc_watershed.py -i <input_dir> -o <output_dir> --save-tiff
```

**Predict Boundaries**:
```bash
python src/postprocessing/predict_boundary.py --img_path <image_path> --mask_path <mask_path> --merge_axes z
```

#### 4. Evaluation and Analysis

**Basic Evaluation**:
```bash
python src/evaluation/evaluate_res.py --gt_file <gt_file> --pred_file <pred_file>
```

**FP/FN 3D Analysis**:
```bash
python src/evaluation/fp_fn_3d_analysis.py \
    --csv_file matches.csv \
    --gt_mask gt_mask.tiff \
    --pred_mask pred_mask.tiff \
    --img image.tiff \
    --output_pdf fp_fn_analysis.pdf
```

#### 5. Visualization

**3D Visualization**:
```bash
python src/visualization/mask_visulize_3d.py --input <mask_file> --output <output_dir>
```

## Main Functional Modules

### Data Processing (src/data/)
- **Format Conversion**: H5 ↔ TIFF, Zarr → TIFF
- **Size Adjustment**: Support for 3D image and mask resizing
- **Data Preprocessing**: Prepare data for nnUNet

### Model Training (src/training/)
- **Auto Training**: Complete nnUNet training pipeline
- **MitoNet**: Empanada-based mitochondria segmentation model
- **Fine-tuning**: Model fine-tuning functionality

### Post-processing (src/postprocessing/)
- **Boundary Generation**: Generate boundary labels for training
- **Instance Segmentation**: BC Watershed algorithm
- **Boundary Prediction**: Use SAM to predict boundaries
- **Instance Optimization**: Optimize instances based on binary segmentation

### Evaluation and Analysis (src/evaluation/)
- **Performance Evaluation**: Calculate F1, Precision, Recall metrics
- **Error Analysis**: FP/FN analysis and 3D visualization
- **Difficult Samples**: Identify and analyze difficult samples

### Visualization (src/visualization/)
- **3D Rendering**: High-quality 3D visualization using VTK
- **Multi-view**: Support for front, side, top, and isometric views
- **PDF Generation**: Generate analysis reports with multiple cases

### Utility Functions (src/utils/)
- **Data Filtering**: Filter data based on mask ID
- **Prediction Filtering**: Filter predictions based on mask

## Configuration Files

### Environment Configuration
- `mitohard_compatible.yml`: Recommended merged environment configuration
- `empanada_env.yml`: Empanada environment configuration
- `sam.yml`: SAM model configuration
- `sam2_mito.yml`: SAM2 mitochondria configuration

### Model Configuration
- `MitoNet_v1.yaml`: MitoNet model configuration

## Scripts and Automation

### SLURM Scripts (scripts/)
- `submit_auto_training.sl`: Auto training submission script
- `submit_evaluate_res.sl`: Evaluation results submission script
- `train_nnunet.sl`: nnUNet training script

### Example Scripts (examples/)
- `run_training_example.sh`: Training example script

## Detailed Documentation

For more detailed information, please refer to the documentation in the `docs/` directory:

- [Auto Training nnUNet Complete Workflow](docs/README_AUTO_TRAIN.md)
- [Mito Filtering Script Usage](docs/README_FILTER_MITO.md)
- [FP/FN 3D Analysis Tool](docs/README_fp_fn_3d_analysis.md)
- [MitoHard Environment Guide](docs/README_mitohard.md)
- [MitoNet Batch Processing Guide](docs/README_mitoNet_batch.md)

## Requirements

### Required Dependencies
- Python 3.7+
- PyTorch
- CUDA (for GPU training)
- nnUNet
- VTK (for 3D visualization)

### Optional Dependencies
- SAM/SAM2 (for boundary prediction)
- Empanada (for MitoNet)
- Napari (for visualization)

## Troubleshooting

### Common Issues

1. **VTK Import Error**: Use `--use_matplotlib` parameter
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **nnUNet Command Not Found**: Ensure nnUNet is properly installed
4. **Dependency Conflicts**: Use the recommended conda environment configuration

### Debugging Suggestions

1. Check log files (`logs/` directory)
2. Test with small datasets
3. Verify environment configuration
4. Review detailed documentation

## Contributing

Issues and Pull Requests are welcome to improve this toolkit.

## License

This project is licensed under the MIT License.

## Changelog

- v2.0: Reorganized code structure, merged README documentation
- v1.0: Initial version with complete mitochondria segmentation pipeline