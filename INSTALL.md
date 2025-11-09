# Installation Guide

## Prerequisites

- Python >= 3.8
- CUDA (for GPU support)
- Conda (recommended) or pip

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mitoem2
```

### 2. Create Conda Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate mitoem2
```

### 3. Install the Package

```bash
# Install in development mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### 4. Verify Installation

```bash
python -c "import mitoem2; print(mitoem2.__version__)"
```

### 5. Test Installation

```bash
# Test that imports work
python -c "from mitoem2.configs import load_config; print('Config loading works!')"
python -c "from mitoem2.models.mitonet.model import MitoNetModel; print('Model import works!')"
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. **Ensure the package is installed**:
   ```bash
   pip install -e .
   ```

2. **Check Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **Verify package structure**:
   ```bash
   python -c "import mitoem2; print(mitoem2.__file__)"
   ```

### Missing Dependencies

If you get missing dependency errors:

1. **Install from requirements.txt**:
   ```bash
   pip install -r requirements.txt
   ```

2. **For empanada (MitoNet)**:
   - Ensure empanada is installed and in your Python path
   - The package expects empanada to be available

3. **For micro-sam (MicroSAM)**:
   ```bash
   pip install micro-sam
   ```

4. **For nnUNet**:
   ```bash
   pip install nnunetv2
   ```

### GPU Issues

If GPU is not detected:

1. **Check CUDA installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Verify PyTorch CUDA version**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

## Development Installation

For development, install with additional tools:

```bash
pip install -e ".[dev]"
```

This installs:
- black (code formatting)
- pytest (testing)
- mypy (type checking)
- ruff (linting)

## Uninstallation

```bash
pip uninstall mitoem2
```

## Next Steps

After installation, see the [README.md](README.md) for usage examples and the [Quick Start Guide](README.md#quick-start).
