"""
Checkpoint management utilities.

Provides utilities for saving and loading model checkpoints.
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None,
    checkpoint_name: str = "checkpoint",
    is_best: bool = False,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a model checkpoint.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state (optional).
        epoch: Current epoch number.
        loss: Current loss value.
        metrics: Dictionary of metrics to save.
        checkpoint_dir: Directory to save checkpoint.
        checkpoint_name: Base name for checkpoint file.
        is_best: Whether this is the best model so far.
        additional_info: Additional information to save.

    Returns:
        Path to saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if loss is not None:
        checkpoint["loss"] = loss

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if additional_info is not None:
        checkpoint.update(additional_info)

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pth"
    torch.save(checkpoint, checkpoint_path)

    # Save best model if applicable
    if is_best:
        best_path = checkpoint_dir / f"{checkpoint_name}_best.pth"
        torch.save(checkpoint, best_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into. If None, only returns state dict.
        optimizer: Optimizer to load state into (optional).
        device: Device to load checkpoint on.

    Returns:
        Dictionary containing checkpoint information.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: Path, pattern: str = "*.pth") -> Optional[Path]:
    """
    Get the latest checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory to search.
        pattern: File pattern to match.

    Returns:
        Path to latest checkpoint, or None if none found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None

    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]
