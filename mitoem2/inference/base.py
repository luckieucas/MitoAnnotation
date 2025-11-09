"""
Base inference engine for mitoem2.

Provides a unified interface for inference across all models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np
import torch

from mitoem2.models.base import BaseModel
from mitoem2.utils.logging import get_logger
from mitoem2.utils.device import get_device

logger = get_logger(__name__)


class BaseInferenceEngine(ABC):
    """
    Base inference engine class.

    Provides a unified interface for inference across different models.
    """

    def __init__(
        self,
        model: BaseModel,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model: Model to use for inference.
            device: Device to run inference on.
            config: Inference configuration.
        """
        self.model = model
        self.config = config or {}

        # Setup device
        if device is None:
            device = get_device(use_gpu=self.config.get("use_gpu", True))
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform inference on an image.

        Args:
            image: Input image as numpy array.
            **kwargs: Additional inference parameters.

        Returns:
            Prediction as numpy array.
        """
        pass

    def predict_batch(self, images: list, **kwargs) -> list:
        """
        Perform batch inference.

        Args:
            images: List of input images.
            **kwargs: Additional inference parameters.

        Returns:
            List of predictions.
        """
        return [self.predict(img, **kwargs) for img in images]

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Callable interface for inference."""
        return self.predict(image, **kwargs)
