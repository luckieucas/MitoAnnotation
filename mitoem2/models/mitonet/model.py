"""
MitoNet model wrapper.

MitoNet is based on the empanada framework. This module provides a wrapper
to integrate it with the mitoem2 unified interface.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger
from mitoem2.empanada import config_loaders

logger = get_logger(__name__)


class MitoNetModel(BaseModel):
    """
    MitoNet model wrapper.

    This class wraps the empanada model to provide a unified interface.
    The actual model is loaded from empanada configuration.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MitoNet model.

        Args:
            config_path: Path to empanada model configuration YAML file.
            config: Model configuration dictionary (alternative to config_path).
        """
        super().__init__(config)

        # Import empanada components from local source
        from mitoem2.utils.empanada_imports import get_empanada_config_loaders, get_empanada_napari_utils
        
        read_yaml = config_loaders.read_yaml
        load_config = config_loaders.load_config
        
        utils_module = get_empanada_napari_utils()
        add_new_model = utils_module.add_new_model

        # Load model configuration
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                # Try relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                alt_path = project_root / config_path
                if alt_path.exists():
                    config_path = alt_path
                else:
                    # Try configs directory
                    alt_path = project_root / "configs" / config_path.name
                    if alt_path.exists():
                        config_path = alt_path
                    else:
                        raise FileNotFoundError(f"Model config not found: {config_path}")
            self.model_config = read_yaml(str(config_path))
            self.config["config_path"] = str(config_path)
        elif config and "config_path" in config:
            config_path = Path(config["config_path"])
            if not config_path.exists():
                # Try relative to project root
                project_root = Path(__file__).parent.parent.parent.parent
                alt_path = project_root / config_path
                if alt_path.exists():
                    config_path = alt_path
                else:
                    # Try configs directory
                    alt_path = project_root / "configs" / config_path.name
                    if alt_path.exists():
                        config_path = alt_path
                    else:
                        raise FileNotFoundError(f"Model config not found: {config_path}")
            self.model_config = read_yaml(str(config_path))
        elif config and "model_config" in config:
            self.model_config = config["model_config"]
        else:
            raise ValueError("Either config_path or config with config_path must be provided")

        # The actual model is loaded lazily when needed
        # This is because empanada models are complex and may require specific initialization
        self._model = None
        self._model_name = self.model_config.get("model_name", "MitoNet")

    def _ensure_model_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._model is None:
            # Load model using empanada's model loading mechanism
            # This is typically done through add_new_model or similar
            logger.info(f"Loading MitoNet model: {self._model_name}")
            # Note: Actual model loading may need to be done differently
            # depending on empanada's API
            self._model = self.model_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Note: MitoNet inference is typically done through Engine3d,
        not through direct forward pass. This method is provided for
        compatibility but may not be used in practice.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        self._ensure_model_loaded()
        # MitoNet typically uses Engine3d for inference, not direct forward
        # This is a placeholder implementation
        raise NotImplementedError(
            "MitoNet inference should be done through Engine3d, "
            "not through direct forward pass. Use MitoNetInferenceEngine instead."
        )

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

        # Update model config with checkpoint path
        self.model_config["checkpoint"] = str(checkpoint_path)
        self.config["checkpoint"] = str(checkpoint_path)
        logger.info(f"Loaded MitoNet checkpoint from {checkpoint_path}")

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint.
            additional_info: Additional information to save.
        """
        # MitoNet checkpoints are typically saved by empanada's training process
        # This is a placeholder for compatibility
        logger.warning(
            "MitoNet checkpoints are typically saved by empanada's training process. "
            "This method may not work as expected."
        )
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_config": self.model_config,
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
