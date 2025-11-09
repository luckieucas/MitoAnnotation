"""
MicroSAM inference engine.

Wrapper for micro-sam automatic instance segmentation for MicroSAM inference.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch

from mitoem2.inference.base import BaseInferenceEngine
from mitoem2.models.microsam.model import MicroSAMModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class MicroSAMInferenceEngine(BaseInferenceEngine):
    """
    MicroSAM inference engine.

    Uses micro-sam's automatic instance segmentation for inference.
    """

    def __init__(
        self,
        model: MicroSAMModel,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MicroSAM inference engine.

        Args:
            model: MicroSAM model instance.
            device: Device to run inference on.
            config: Inference configuration.
        """
        super().__init__(model, device, config)

        # Import micro-sam components
        try:
            from micro_sam.automatic_segmentation import automatic_instance_segmentation
            self._automatic_instance_segmentation = automatic_instance_segmentation
        except ImportError as e:
            raise ImportError(
                "micro-sam is required for MicroSAM inference. "
                "Please install it: pip install micro-sam"
            ) from e

        # Configuration
        self.min_size = config.get("min_size", 500) if config else 500
        self.tile_shape = config.get("tile_shape", [512, 512]) if config else [512, 512]
        self.halo = config.get("halo", [16, 16]) if config else [16, 16]
        self.use_embeddings = config.get("use_embeddings", False) if config else False
        self.embedding_path = config.get("embedding_path", None) if config else None

        # Determine whether to use automatic mask generation (AMG)
        requested_amg = bool(config.get("use_amg", False)) if config else False

        # Get predictor and segmenter from model, fallback to AMG if needed
        is_tiled = self.tile_shape is not None
        try:
            self.predictor, self.segmenter = model.get_predictor_and_segmenter(
                device=device,
                amg=requested_amg,
                is_tiled=is_tiled,
            )
            self.use_amg = requested_amg
        except RuntimeError as err:
            if (
                not requested_amg
                and "does not contain a segmentation decoder" in str(err)
            ):
                logger.warning(
                    "Segmentation decoder not available; switching to automatic mask generation (AMG)."
                )
                self.predictor, self.segmenter = model.get_predictor_and_segmenter(
                    device=device,
                    amg=True,
                    is_tiled=is_tiled,
                )
                self.use_amg = True
            else:
                raise

    def predict(self, image: np.ndarray, embedding_path: Optional[str] = None, **kwargs) -> np.ndarray:
        """
        Perform automatic instance segmentation on an image.

        Args:
            image: Input image as numpy array (2D or 3D).
            embedding_path: Optional path to save/load embeddings.
            **kwargs: Additional parameters.

        Returns:
            Segmentation as numpy array with instance labels.
        """
        if image.ndim not in [2, 3]:
            raise ValueError(f"Input image must be 2D or 3D, but has {image.ndim} dimensions")

        # Use provided embedding_path or fall back to config
        if embedding_path is None:
            embedding_path = self.embedding_path

        logger.info(f"Running automatic instance segmentation on {image.ndim}D image with shape {image.shape}")

        # Run automatic instance segmentation
        prediction = self._automatic_instance_segmentation(
            predictor=self.predictor,
            segmenter=self.segmenter,
            input_path=image,  # Can be numpy array or path
            embedding_path=embedding_path,
            ndim=image.ndim,
            tile_shape=tuple(self.tile_shape) if self.tile_shape else None,
            halo=tuple(self.halo) if self.halo else None,
            verbose=True
        )

        # Post-process segmentation (filter small instances and relabel)
        prediction = self._postprocess_segmentation(prediction)

        return prediction

    def _postprocess_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Post-process segmentation by filtering out small instances and relabeling.
        
        Args:
            segmentation: Input segmentation mask with instance labels
            
        Returns:
            Processed segmentation with small instances removed and consecutive labels
        """
        from skimage.segmentation import relabel_sequential
        from skimage import measure
        
        logger.info(f"Post-processing segmentation...")
        logger.info(f"  Original number of instances (including background): {len(np.unique(segmentation))}")
        
        # Filter out labels with size smaller than min_size
        logger.info(f"  Filtering instances smaller than {self.min_size} voxels...")
        filtered_mask = segmentation.copy()
        unique_labels, counts = np.unique(filtered_mask, return_counts=True)
        small_labels = unique_labels[(counts < self.min_size) & (unique_labels != 0)]
        
        if small_labels.size > 0:
            logger.info(f"  Removing {len(small_labels)} small instances.")
            filtered_mask[np.isin(filtered_mask, small_labels)] = 0
        else:
            logger.info("  No small instances to remove.")
        
        # Relabel all remaining objects to be consecutive from 1
        logger.info("  Relabeling to ensure consecutive numbering...")
        relabeled_mask, num_labels = measure.label(filtered_mask, background=0, return_num=True)
        logger.info(f"  Final number of instances (excluding background): {num_labels}")
        
        return relabeled_mask.astype(np.uint32)
