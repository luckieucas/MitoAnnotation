"""
MitoNet inference engine.

Wrapper for empanada's Engine3d for MitoNet inference.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch

from mitoem2.inference.base import BaseInferenceEngine
from mitoem2.models.mitonet.model import MitoNetModel
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class MitoNetInferenceEngine(BaseInferenceEngine):
    """
    MitoNet inference engine.

    Uses empanada's Engine3d for 3D inference.
    """

    def __init__(
        self,
        model: MitoNetModel,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MitoNet inference engine.

        Args:
            model: MitoNet model instance.
            device: Device to run inference on.
            config: Inference configuration.
        """
        # Import empanada components from local source
        from mitoem2.utils.empanada_imports import get_empanada_napari_inference
        
        inference_module = get_empanada_napari_inference()
        Engine3d = inference_module.Engine3d
        self.tracker_consensus = inference_module.tracker_consensus
        self.stack_postprocessing = inference_module.stack_postprocessing

        super().__init__(model, device, config)

        # Initialize empanada engine
        model_config = model.model_config
        self.engine = Engine3d(
            model_config,
            inference_scale=config.get("downsampling", 1),
            confidence_thr=config.get("confidence_thr", 0.5),
            nms_threshold=config.get("center_confidence_thr", 0.1),
            nms_kernel=config.get("min_distance_object_centers", 3),
            min_size=config.get("min_size", 500),
            min_extent=config.get("min_extent", 5),
            use_gpu=config.get("use_gpu", True),
        )

        self.axes = config.get("axes", ["xy", "xz", "yz"])

    def predict(
        self, 
        image: np.ndarray, 
        axes: Optional[List[str]] = None,
        save_axis_results: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Perform 3D inference on an image.

        Args:
            image: Input 3D image as numpy array.
            axes: List of axes to perform inference on (default: ["xy", "xz", "yz"]).
            save_axis_results: If True, stores axis results in self.axis_results.
            **kwargs: Additional parameters.

        Returns:
            3D segmentation as numpy array (consensus result).
        """
        if axes is None:
            axes = self.axes

        if image.ndim != 3:
            raise ValueError(f"Expected 3D image, got {image.ndim}D")

        logger.info(f"Running MitoNet inference on axes: {axes}")

        # Run inference on each axis
        trackers = {}
        for axis in axes:
            logger.info(f"Processing {axis} plane...")
            _, tracker = self.engine.infer_on_axis(image, axis)
            trackers[axis] = tracker

        # Optionally save per-axis results
        if save_axis_results:
            self.axis_results = {}
            model_config = self.model.model_config
            
            for axis in axes:
                logger.info(f"Post-processing {axis} plane results...")
                axis_postprocess_worker = self.stack_postprocessing(
                    {axis: trackers[axis]},
                    store_url=None,
                    model_config=model_config,
                    min_size=self.config.get("min_size", 500),
                    min_extent=self.config.get("min_extent", 5),
                )
                
                axis_segmentation = None
                for seg_vol, class_name, _ in axis_postprocess_worker:
                    logger.info(f"Generated {axis} segmentation for class '{class_name}'")
                    if axis_segmentation is None:
                        axis_segmentation = seg_vol
                
                if axis_segmentation is not None:
                    self.axis_results[axis] = axis_segmentation
                else:
                    logger.warning(f"Could not generate segmentation for {axis} plane")

        # Post-process and combine results using tracker_consensus
        # This combines predictions from multiple axes (xy, xz, yz)
        model_config = self.model.model_config
        pixel_vote_thr = self.config.get("pixel_vote_thr", 2)
        
        consensus_worker = self.tracker_consensus(
            trackers,
            store_url=None,
            model_config=model_config,
            pixel_vote_thr=pixel_vote_thr,
            min_size=self.config.get("min_size", 500),
            min_extent=self.config.get("min_extent", 5),
        )

        # tracker_consensus yields (consensus_vol, class_name, instances) tuples
        segmentation = None
        for consensus_vol, class_name, _ in consensus_worker:
            logger.info(f"Generated consensus for class '{class_name}'")
            if segmentation is None:
                segmentation = consensus_vol
            else:
                # If multiple classes, combine them (for now, just take the first)
                # TODO: Handle multi-class segmentation properly
                logger.warning(f"Multiple classes detected, using first class: {class_name}")

        if segmentation is None:
            raise RuntimeError("Failed to generate segmentation")

        return segmentation
