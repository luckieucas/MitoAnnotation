# Configuration Files Guide

This directory contains YAML configuration files for different workflows in MitoAnnotation.

## Quick Start

### Using Default Configurations

```bash
# Run MitoNet baseline on dataset 1
just mitonet_baseline 1

# Run MicroSAM baseline on dataset 2
just microsam_baseline 2

# Run MitoNet fine-tuning on dataset 3
just mitonet_finetune 3
```

### Using Custom Configurations

1. **Create a custom config file** (e.g., `my_config.yaml`):

```yaml
dataset: "2"
eval: true
use_gpu: true
min_size: 1000  # Custom value
```

2. **Run with custom config**:

```bash
# Use custom config file
just mitonet_baseline configs/my_config.yaml

# Or override specific parameters
just mitonet_baseline 2 configs/my_config.yaml
```

## Configuration Files

### `mitonet_baseline_default.yaml`
Default configuration for MitoNet baseline inference.

**Key Parameters:**
- `dataset`: Dataset ID or name
- `use_gpu`: Enable/disable GPU usage
- `axes`: Inference planes (xy, xz, yz)
- `min_size`: Minimum instance size (voxels)
- `eval`: Enable evaluation mode

### `microsam_baseline_default.yaml`
Default configuration for MicroSAM baseline inference.

**Key Parameters:**
- `dataset`: Dataset ID or name
- `mode`: "ZS" (zero-shot) or "FT" (fine-tuned)
- `model_type`: SAM model variant
- `tile_shape`: Tile size for processing
- `eval`: Enable evaluation mode

### `mitonet_finetune_default.yaml`
Default configuration for MitoNet fine-tuning.

**Key Parameters:**
- `dataset`: Dataset ID or name
- `iterations`: Number of training iterations
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `test_after_training`: Auto-test after training

## Creating Dataset-Specific Configs

For convenience, you can create dataset-specific configs:

```bash
# Create configs/dataset2.yaml
cp configs/mitonet_baseline_default.yaml configs/dataset2.yaml
# Edit dataset2.yaml to set dataset: "2" and other custom parameters

# Then run:
just mitonet_baseline configs/dataset2.yaml
```

## Parameter Priority

When both config file and command-line arguments are provided:

1. **Command-line arguments** override config file values
2. **Config file values** override defaults
3. **Default values** are used if not specified

Example:
```bash
# Config file has: dataset: "1", min_size: 500
# Command line: just mitonet_baseline 2
# Result: Uses dataset "2" (overridden), min_size 500 (from config)
```

## Submitting to Slurm

All launch commands now support config files:

```bash
# Use default config
just launch_mitonet_baseline 2

# Use custom config with custom time/partition
just launch_mitonet_baseline configs/my_config.yaml "12:00:00" "short"
```

## Tips

1. **Start with defaults**: Use the default configs as templates
2. **Create per-dataset configs**: Helpful for experiments on multiple datasets
3. **Version control your configs**: Track experiment parameters
4. **Comment your changes**: Add comments in YAML files to document custom settings

## Example Workflow

```bash
# 1. Create a custom config for dataset 5
cat > configs/dataset5_experiment.yaml << EOF
dataset: "5"
eval: true
use_gpu: true
min_size: 800
confidence_thr: 0.6
EOF

# 2. Run locally to test
just mitonet_baseline configs/dataset5_experiment.yaml

# 3. Submit to cluster
just launch_mitonet_baseline configs/dataset5_experiment.yaml "12:00:00" "weilab"
```




