"""
nnUNet model wrapper.

nnUNet is a separate framework. This module provides a wrapper to integrate
it with the mitoem2 unified interface.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class nnUNetModel(BaseModel):
    """
    nnUNet model wrapper.

    Note: nnUNet is typically used through its own command-line interface.
    This wrapper provides basic integration but most operations should
    be done through nnUNet's native tools.
    """

    def __init__(
        self,
        dataset_id: int,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize nnUNet model.

        Args:
            dataset_id: nnUNet dataset ID.
            config: Model configuration dictionary.
        """
        super().__init__(config)

        try:
            from nnunetv2.paths import nnUNet_results
            from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
        except ImportError as e:
            raise ImportError(
                "nnunetv2 is required for nnUNet. "
                "Please install it: pip install nnunetv2"
            ) from e

        self.dataset_id = dataset_id
        self.dataset_name = maybe_convert_to_dataset_name(dataset_id)
        self.config["dataset_id"] = dataset_id
        self.config["dataset_name"] = self.dataset_name

        logger.info(f"Initialized nnUNet model for dataset: {self.dataset_name} (ID: {dataset_id})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Note: nnUNet inference is typically done through its command-line
        interface or predictor API, not through direct forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError(
            "nnUNet inference should be done through nnUNet's native tools. "
            "Use nnUNetInferenceEngine or nnunetv2_predict command instead."
        )

    def load_weights(self, checkpoint_path: Path, strict: bool = True) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory (nnUNet stores models in directories).
            strict: Whether to strictly enforce that the keys match.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        self.config["checkpoint_path"] = str(checkpoint_path)
        logger.info(f"Set nnUNet checkpoint path to {checkpoint_path}")

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to checkpoint.

        Note: nnUNet models are typically saved by its training process.
        This method is a placeholder for compatibility.

        Args:
            checkpoint_path: Path to save checkpoint directory.
            additional_info: Additional information to save.
        """
        logger.warning(
            "nnUNet models are typically saved by its training process. "
            "This method may not work as expected."
        )
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        # Save as metadata file
        import json
        with open(checkpoint_path / "model_info.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
