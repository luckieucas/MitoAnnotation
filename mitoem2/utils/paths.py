"""
Path management utilities.

This module provides centralized path management to avoid hardcoded paths
and sys.path modifications throughout the codebase.
"""
import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root directory.
    """
    # This file is in mitoem2/utils/paths.py
    # Project root is two levels up
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent


def get_data_root() -> Path:
    """
    Get the default data root directory.

    Returns:
        Path to the data root directory. Defaults to nnUNet_raw if available,
        otherwise returns a data directory in the project root.
    """
    # Try to use nnUNet_raw if available
    try:
        from nnunetv2.paths import nnUNet_raw
        if nnUNet_raw and os.path.exists(nnUNet_raw):
            return Path(nnUNet_raw)
    except (ImportError, AttributeError):
        pass

    # Fallback to project data directory
    project_root = get_project_root()
    return project_root / "data"


def get_checkpoint_dir(checkpoint_name: Optional[str] = None) -> Path:
    """
    Get the checkpoint directory.

    Args:
        checkpoint_name: Optional checkpoint name. If None, returns the base checkpoint directory.

    Returns:
        Path to the checkpoint directory.
    """
    project_root = get_project_root()
    checkpoint_base = project_root / "checkpoints"
    
    if checkpoint_name:
        return checkpoint_base / checkpoint_name
    return checkpoint_base


def get_log_dir() -> Path:
    """
    Get the log directory.

    Returns:
        Path to the log directory.
    """
    project_root = get_project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_config_dir() -> Path:
    """
    Get the configuration directory.

    Returns:
        Path to the configuration directory.
    """
    # First check for user configs in project root
    project_root = get_project_root()
    user_config_dir = project_root / "configs"
    if user_config_dir.exists():
        return user_config_dir
    
    # Otherwise use package configs
    package_root = Path(__file__).parent.parent
    return package_root / "configs"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        The same path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
