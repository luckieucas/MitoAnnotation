# MitoEM2 - Mitochondria Segmentation Toolkit

A comprehensive, production-ready toolkit for mitochondria segmentation in EM images. Supports three state-of-the-art methods: **MitoNet**, **MicroSAM**, and **nnUNet**.

## Features

- **Unified Interface**: Consistent API for all three segmentation methods
- **Configuration Management**: YAML-based configuration with dataclass validation
- **Production-Ready**: Logging, checkpoint management, multi-GPU support, early stopping
- **Extensible**: Easy to add new models or datasets

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd mitoem2

# Install in development mode
pip install -e .

# Or install with conda
conda env create -f environment.yml
conda activate mitoem2
pip install -e .
```

### Dependencies

The project requires:
- Python >= 3.8
- PyTorch >= 1.8
- See  `environment.yml` for full dependencies

## Quick Start

### 1. Install the Package

```bash
pip install -e .
```

### 2. Train a Model

```bash
# Train MitoNet
python -m mitoem2.scripts.train --method mitonet --config configs/mitonet/train_default.yaml --dataset 1

# Train MicroSAM
python -m mitoem2.scripts.train --method microsam --config configs/microsam/train_default.yaml --dataset 1

# Or use the command-line entry points (after installation)
mitoem2-train --method mitonet --config configs/mitonet/train_default.yaml --dataset 1
```

### 3. Run Inference

```bash
# Run inference
python -m mitoem2.scripts.inference --method mitonet --input /path/to/image.tiff --output /path/to/output

# Or use command-line entry point
mitoem2-inference --method mitonet --input /path/to/image.tiff --output /path/to/output
```

### 4. Evaluate Results

```bash
# Evaluate predictions
python -m mitoem2.scripts.evaluate --pred /path/to/prediction.tiff --gt /path/to/ground_truth.tiff

# Or use command-line entry point
mitoem2-evaluate --pred /path/to/prediction.tiff --gt /path/to/ground_truth.tiff
```

## Project Structure

```
mitoem2/
├── mitoem2/              # Main package (installable)
│   ├── configs/         # Configuration management
│   ├── data/             # Data loading and processing
│   ├── models/           # Model definitions
│   ├── training/         # Training modules
│   ├── inference/        # Inference engines
│   ├── evaluation/       # Evaluation metrics
│   ├── utils/            # Utility functions
│   └── scripts/          # Command-line scripts
├── configs/              # User configuration files
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
├── src/                  # Legacy code (for backward compatibility)
├── setup.py              # Package setup
├── requirements.txt      # Python dependencies
└── environment.yml       # Conda environment
```

## Configuration

### Using YAML Configuration Files

Create a configuration file (e.g., `my_config.yaml`):

```yaml
method: mitonet
dataset:
  id: 1
  name: null

model:
  config_path: configs/MitoNet_v1.yaml
  checkpoint: null

training:
  iterations: 2000
  batch_size: 16
  learning_rate: 0.003

inference:
  use_gpu: true
  confidence_thr: 0.5
  min_size: 500

logging:
  level: INFO
  use_wandb: false
```

Then use it:

```bash
python -m mitoem2.scripts.train --config my_config.yaml
```

### Command-Line Overrides

You can override any configuration value from the command line:

```bash
python -m mitoem2.scripts.train --config my_config.yaml --dataset 2 --output_dir ./my_checkpoints
```

## Methods

### MitoNet

Based on the empanada framework. Best for 3D segmentation with multi-view inference.

```bash
# Training
python -m mitoem2.scripts.train --method mitonet --dataset 1

# Inference
python -m mitoem2.scripts.inference --method mitonet --input image.tiff --output output_dir
```

### MicroSAM

Based on Segment Anything Model (SAM). Best for zero-shot and few-shot segmentation.

```bash
# Training (fine-tuning)
python -m mitoem2.scripts.train --method microsam --dataset 1

# Inference
python -m mitoem2.scripts.inference --method microsam --input image.tiff --output output_dir
```

### nnUNet

Based on nnUNet framework. Best for large-scale training with automatic preprocessing.

```bash
# Training (uses nnUNet's native command)
nnunetv2_train 6 3d_fullres 0

# Inference (uses nnUNet's native command)
nnunetv2_predict -i input_folder -o output_folder -d 6 -f 0
```

## Data Format

### nnUNet Format

The toolkit expects data in nnUNet format:

```
Dataset001_MitoHardBeta/
├── imagesTr/
│   └── case_0000.nii.gz
├── instancesTr/
│   └── case.nii.gz
├── imagesTs/
└── instancesTs/
```

### Empanada Format (for MitoNet)

For MitoNet training, data is automatically converted to 2D slice format:

```
data/
├── train/
│   └── volume_name/
│       ├── images/
│       │   └── slice_*.tif
│       └── masks/
│           └── slice_*.tif
└── val/
    └── volume_name/
        ├── images/
        └── masks/
```

## API Usage

### Python API

```python
from mitoem2.configs import load_config
from mitoem2.models.mitonet.model import MitoNetModel
from mitoem2.inference.mitonet_inference import MitoNetInferenceEngine
import numpy as np

# Load configuration
config = load_config("configs/mitonet/inference_default.yaml")

# Load model
model = MitoNetModel(config_path=config.model.config_path)
if config.model.checkpoint:
    model.load_weights(config.model.checkpoint)

# Create inference engine
engine = MitoNetInferenceEngine(
    model=model,
    config=config.inference.__dict__,
)

# Run inference
image = np.load("image.npy")  # Your 3D image
segmentation = engine.predict(image)
```

## Advanced Features

### Multi-GPU Training

```python
from mitoem2.training.trainer import BaseTrainer

trainer = BaseTrainer(
    model=model,
    train_loader=train_loader,
    use_multi_gpu=True,  # Enable multi-GPU
)
```

### Wandb Integration

```yaml
logging:
  use_wandb: true
  wandb_project: my_project
```

### Early Stopping

```python
from mitoem2.training.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)
```

## Migration from Legacy Code

The new package structure is backward compatible. Legacy code in `src/` can still be used, but we recommend migrating to the new API:

### Old Way (with sys.path)

```python
import sys
sys.path.append('/path/to/MitoAnnotation/src')
from empanada_napari.inference import Engine3d
```

### New Way (with package)

```python
from mitoem2.inference.mitonet_inference import MitoNetInferenceEngine
```

## Contributing

Contributions are welcome! Please ensure your code follows the project's style guidelines and includes appropriate tests.

## License

[Add your license here]

## Citation

If you use this toolkit in your research, please cite:

```bibtex
[Add citation information]
```

## Acknowledgments

- **empanada**: For the MitoNet framework
- **micro-sam**: For the MicroSAM implementation
- **nnUNet**: For the nnUNet framework

## Support

For issues and questions, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a refactored version of the original project. The legacy code in `src/` is maintained for backward compatibility but is deprecated in favor of the new `mitoem2` package structure.