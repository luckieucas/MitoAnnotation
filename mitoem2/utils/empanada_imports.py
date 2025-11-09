"""
Utilities for importing empanada from local source code.

This module ensures that empanada and empanada_napari can be imported
from the mitoem2 package directory without requiring installation.
"""
from pathlib import Path

# Get mitoem2 package directory (mitoem2/utils/empanada_imports.py -> mitoem2/utils -> mitoem2)
_mitoem2_package_dir = Path(__file__).parent.parent

# Try to import from mitoem2 package directory first (preferred)
# If not found, fall back to src/ directory
_empanada_dir = _mitoem2_package_dir / "empanada"
_empanada_napari_dir = _mitoem2_package_dir / "empanada_napari"

# Fallback to src directory if not in package
if not _empanada_dir.exists():
    _project_root = _mitoem2_package_dir.parent
    _src_dir = _project_root / "src"
    _empanada_dir = _src_dir / "empanada"
    _empanada_napari_dir = _src_dir / "empanada_napari"
    _use_src = True  # Use src directory
else:
    _use_src = False  # Use mitoem2 package (preferred)

# Verify that empanada directories exist
if not _empanada_dir.exists() or not _empanada_napari_dir.exists():
    raise ImportError(
        f"empanada directories not found. "
        f"Expected: {_empanada_dir} and {_empanada_napari_dir}. "
        f"Please ensure empanada and empanada_napari are in mitoem2/ or src/ directory."
    )


def import_empanada():
    """
    Import empanada modules from local source.
    
    Returns:
        Tuple of (empanada, empanada_napari) modules.
    """
    try:
        import empanada
        import empanada_napari
        return empanada, empanada_napari
    except ImportError as e:
        raise ImportError(
            f"Could not import empanada from {_src_dir}. "
            f"Please ensure src/empanada and src/empanada_napari exist."
        ) from e


def get_empanada_config_loaders():
    """Get empanada config_loaders module."""
    if not _use_src:
        # Import from mitoem2 package (preferred) - use mitoem2.empanada
        from mitoem2 import empanada
    else:
        # Import from src directory (fallback)
        import sys
        _src_dir = _mitoem2_package_dir.parent / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        import empanada
    
    return empanada.config_loaders


def get_empanada_napari_finetune():
    """Get empanada_napari finetune module."""
    if not _use_src:
        # Import from mitoem2 package (preferred) - use mitoem2.empanada_napari.finetune
        from mitoem2.empanada_napari import finetune
        return finetune
    else:
        # Import from src directory (fallback)
        import sys
        _src_dir = _mitoem2_package_dir.parent / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        from empanada_napari import finetune
        return finetune


def get_empanada_napari_inference():
    """Get empanada_napari inference module."""
    if not _use_src:
        # Import from mitoem2 package (preferred) - use mitoem2.empanada_napari.inference
        from mitoem2.empanada_napari import inference
        return inference
    else:
        # Import from src directory (fallback)
        import sys
        _src_dir = _mitoem2_package_dir.parent / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        from empanada_napari import inference
        return inference


def get_empanada_napari_utils():
    """Get empanada_napari utils module."""
    if not _use_src:
        # Import from mitoem2 package (preferred) - use mitoem2.empanada_napari.utils
        from mitoem2.empanada_napari import utils
        return utils
    else:
        # Import from src directory (fallback)
        import sys
        _src_dir = _mitoem2_package_dir.parent / "src"
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        from empanada_napari import utils
        return utils
