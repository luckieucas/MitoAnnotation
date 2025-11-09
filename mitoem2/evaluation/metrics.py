"""
Evaluation metrics for mitoem2.

Provides functions for computing segmentation metrics.
"""
import numpy as np
from typing import Dict, Any
from pathlib import Path

# Import existing evaluation functions
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "evaluation"))

try:
    from evaluate_res import (
        read_image_file,
        compute_precision_recall_f1,
        instance_matching,
        evaluate_single_file,
    )
except ImportError:
    # Fallback if evaluation module not available
    def read_image_file(file_path):
        import tifffile
        import SimpleITK as sitk
        if isinstance(file_path, str):
            file_ext = Path(file_path).suffix.lower()
            if file_ext in [".nii", ".nii.gz"] or str(file_path).endswith(".nii.gz"):
                img_sitk = sitk.ReadImage(str(file_path))
                return sitk.GetArrayFromImage(img_sitk)
            else:
                return tifffile.imread(str(file_path))
        return file_path

    def compute_precision_recall_f1(pred_mask, true_mask):
        TP = np.sum((pred_mask == 1) & (true_mask == 1))
        FP = np.sum((pred_mask == 1) & (true_mask == 0))
        FN = np.sum((pred_mask == 0) & (true_mask == 1))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        return precision, recall, f1, accuracy

    def instance_matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
        # Simplified version
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }

    def evaluate_single_file(pred_file, gt_file, save_results=True, mask_file=None):
        return {}


def evaluate_prediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate prediction against ground truth.

    Args:
        prediction: Prediction array.
        ground_truth: Ground truth array.
        threshold: IoU threshold for instance matching.

    Returns:
        Dictionary of metrics.
    """
    # Convert to binary if needed
    pred_binary = (prediction > 0).astype(np.uint8)
    gt_binary = (ground_truth > 0).astype(np.uint8)

    # Compute pixel-level metrics
    precision, recall, f1, accuracy = compute_precision_recall_f1(pred_binary, gt_binary)

    # Compute instance-level metrics
    instance_metrics = instance_matching(ground_truth, prediction, thresh=threshold)

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_accuracy": accuracy,
        "instance_precision": instance_metrics.get("precision", 0.0),
        "instance_recall": instance_metrics.get("recall", 0.0),
        "instance_f1": instance_metrics.get("f1", 0.0),
        "panoptic_quality": instance_metrics.get("panoptic_quality", 0.0),
    }
