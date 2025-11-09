"""
Device management utilities.

Provides utilities for managing GPU/CPU devices in a unified way.
"""
import torch
from typing import Optional, Union


def get_device(use_gpu: bool = True, device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device (GPU or CPU).

    Args:
        use_gpu: Whether to try to use GPU.
        device_id: Specific GPU device ID. If None, uses default GPU.

    Returns:
        PyTorch device object.
    """
    if use_gpu and torch.cuda.is_available():
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Device ID {device_id} not available. "
                    f"Only {torch.cuda.device_count()} GPU(s) available."
                )
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def get_available_gpus() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of available GPUs.
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_gpu_available() -> bool:
    """
    Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise.
    """
    return torch.cuda.is_available()
