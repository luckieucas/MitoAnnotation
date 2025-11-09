"""
Base model class for mitoem2.

Provides a common interface for all models used in the project.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Base model class for all mitoem2 models.

    This class provides a common interface for different model types,
    ensuring consistent behavior across MitoNet, MicroSAM, and nnUNet models.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        pass

    def load_weights(self, checkpoint_path: Path, strict: bool = True) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            strict: Whether to strictly enforce that the keys match.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=strict)

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint.
            additional_info: Additional information to save with checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)

    def get_num_parameters(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (override in subclasses if needed)."""
        pass

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (override in subclasses if needed)."""
        pass
