"""
MicroSAM model wrapper.

MicroSAM is based on the Segment Anything Model (SAM). This module provides
a wrapper to integrate it with the mitoem2 unified interface.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class MicroSAMModel(BaseModel):
    """
    MicroSAM model wrapper.

    This class wraps the micro-sam model to provide a unified interface.
    """

    def __init__(
        self,
        model_type: str = "vit_b_em_organelles",
        checkpoint_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MicroSAM model.

        Args:
            model_type: SAM model type (e.g., "vit_b_em_organelles").
            checkpoint_path: Optional path to custom checkpoint.
            config: Model configuration dictionary.
        """
        super().__init__(config)

        # Import micro-sam components
        try:
            from micro_sam.automatic_segmentation import get_predictor_and_segmenter
            from micro_sam.util import get_model_names
        except ImportError as e:
            raise ImportError(
                "micro-sam is required for MicroSAM. "
                "Please install it: pip install micro-sam"
            ) from e

        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.config["model_type"] = model_type
        if checkpoint_path:
            self.config["checkpoint_path"] = str(checkpoint_path)

        # Store imports for later use
        self._get_predictor_and_segmenter = get_predictor_and_segmenter
        self._get_model_names = get_model_names

        # Don't load model here - load lazily when needed
        # MicroSAM uses predictor/segmenter, not direct model access
        self._model = None
        self._predictor = None
        self._segmenter = None
        
        # Validate model type
        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Validate that the model type is available."""
        try:
            available_models = self._get_model_names()
            if self.model_type not in available_models:
                logger.warning(
                    f"Model type '{self.model_type}' not in available models: {available_models}. "
                    f"Using it anyway, but it may fail."
                )
        except Exception as e:
            logger.warning(f"Could not validate model type: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Note: MicroSAM inference typically uses the predictor API,
        not direct forward pass. This method is provided for compatibility.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # MicroSAM doesn't support direct forward pass
        # Use get_predictor() or get_predictor_and_segmenter() instead
        raise NotImplementedError(
            "MicroSAM doesn't support direct forward pass. "
            "Use get_predictor() or get_predictor_and_segmenter() for inference."
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

        self.checkpoint_path = checkpoint_path
        self.config["checkpoint_path"] = str(checkpoint_path)
        # Reset predictor/segmenter so they will be reloaded with new checkpoint
        self._predictor = None
        self._segmenter = None
        logger.info(f"MicroSAM checkpoint path set to {checkpoint_path}. Will be used when creating predictor/segmenter.")

    def save_weights(self, checkpoint_path: Path, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to checkpoint.

        Note: MicroSAM models are typically saved through the training process.
        This method is a placeholder for compatibility.

        Args:
            checkpoint_path: Path to save checkpoint.
            additional_info: Additional information to save.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # MicroSAM doesn't expose the underlying model directly
        # Save configuration and checkpoint path reference instead
        checkpoint = {
            "model_type": self.model_type,
            "config": self.config,
        }
        
        if self.checkpoint_path:
            checkpoint["checkpoint_path"] = str(self.checkpoint_path)

        if additional_info:
            checkpoint.update(additional_info)

        import json
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Saved MicroSAM configuration to {checkpoint_path}")
        logger.warning("Note: MicroSAM model weights should be saved through the training process.")

    def get_predictor(self):
        """
        Get the SAM predictor for inference.

        Returns:
            SAM predictor instance.
        """
        if self._model is None:
            self._load_model()

        try:
            from micro_sam.predictor import SamPredictor
            return SamPredictor(self._model)
        except ImportError:
            from segment_anything import SamPredictor
            return SamPredictor(self._model)
    
    def get_predictor_and_segmenter(self, device=None, amg=False, is_tiled=False):
        """
        Get predictor and segmenter for automatic instance segmentation.
        
        Args:
            device: Device to use (default: auto-detect).
            amg: Whether to use AMG (Automatic Mask Generation).
            is_tiled: Whether to use tiled processing.
            
        Returns:
            Tuple of (predictor, segmenter).
        """
        # Use cached predictor/segmenter if available
        if self._predictor is not None and self._segmenter is not None:
            return self._predictor, self._segmenter
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = str(self.checkpoint_path) if self.checkpoint_path else None
        predictor, segmenter = self._get_predictor_and_segmenter(
            model_type=self.model_type,
            checkpoint=checkpoint,
            device=device,
            amg=amg,
            is_tiled=is_tiled,
        )
        
        # Cache predictor and segmenter
        self._predictor = predictor
        self._segmenter = segmenter
        
        return predictor, segmenter
