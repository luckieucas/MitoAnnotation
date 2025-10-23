from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
import numpy as np
import argparse
import tifffile as tiff
import os
import glob
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from scipy import ndimage
from scipy.ndimage import label
import multiprocessing
import pandas as pd
from skimage.segmentation import relabel_sequential
from connectomics.utils.evaluate import _check_label_array, _raise, matching_criteria, label_overlap
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Import iou_tracking modules
from nnunetv2.postprocessing.iou_tracking import binary_volume_to_instances_iou_tracking


def optimize_parallel_settings(num_processes=None, batch_size=None):
    """
    Optimize parallel processing settings based on system resources
    
    Args:
        num_processes: user-specified number of processes (None for auto)
        batch_size: user-specified batch size (None for auto)
        
    Returns:
        tuple: (optimized_num_processes, optimized_batch_size)
    """
    # Get system information
    cpu_count = multiprocessing.cpu_count()
    gpu_available = torch.cuda.is_available()
    
    # Optimize number of processes
    if num_processes is None:
        if gpu_available:
            # For GPU inference, use fewer processes to avoid GPU memory conflicts
            num_processes = min(4, cpu_count // 2)
        else:
            # For CPU inference, can use more processes
            num_processes = min(6, cpu_count - 1)
    else:
        num_processes = min(num_processes, cpu_count)
    
    # Optimize batch size
    if batch_size is None:
        if gpu_available:
            # For GPU, use larger batches to better utilize GPU memory
            batch_size = 32
        else:
            # For CPU, use smaller batches to manage memory
            batch_size = 16
    else:
        batch_size = max(1, batch_size)
    
    print(f"System info: {cpu_count} CPUs, GPU: {gpu_available}")
    print(f"Optimized settings: {num_processes} processes, batch size: {batch_size}")
    
    return num_processes, batch_size


def load_image(file_path):
    """
    Load image from different formats (tiff, nii.gz) using appropriate readers
    """
    file_path = str(file_path)
    if file_path.endswith(('.nii.gz', '.nii')):
        # Load nii.gz file using nnUNet's SimpleITKIO
        io = SimpleITKIO()
        img_data, properties = io.read_images([file_path])
        img_data = img_data[0]  # Get the first (and only) image
        # Ensure correct data type and orientation
        img_data = np.array(img_data, dtype=np.float32)
        print(f"Loaded nii.gz: {file_path}, shape: {img_data.shape}, dtype: {img_data.dtype}")
        return img_data
    elif file_path.endswith(('.tif', '.tiff')):
        # Load tiff file
        img_data = tiff.imread(file_path)
        print(f"Loaded tiff: {file_path}, shape: {img_data.shape}, dtype: {img_data.dtype}")
        return img_data
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_prediction(pred, save_path, input_format):
    """
    Save prediction in the same format as input using appropriate writers
    """
    save_path = str(save_path)
    if input_format in ['.nii.gz', '.nii']:
        # Save nii.gz file using nnUNet's SimpleITKIO
        io = SimpleITKIO()
        # Create properties dictionary for the prediction
        properties = {
            'spacing': [1.0, 1.0, 1.0],
            'origin': [0.0, 0.0, 0.0],
            'direction': np.eye(3).flatten()
        }
        # Save the prediction
        io.write_seg(pred, properties, save_path)
        print(f"Saved prediction to: {save_path}")
    elif input_format in ['.tif', '.tiff']:
        # Save as tiff
        if pred.max() > 256:
            pred = pred.astype(np.uint16)
        else:
            pred = pred.astype(np.uint8)
        tiff.imwrite(save_path, pred, compression="zlib")
        print(f"Saved prediction to: {save_path}")
    else:
        raise ValueError(f"Cannot save format: {input_format}")

def apply_iou_tracking(prediction, axis='xy', iou_threshold=0.3, ioa_threshold=0.5, 
                       min_size=500, min_extent=3, max_size=None, 
                       remove_border_instances=False, max_aspect_ratio=None):
    """
    Apply IoU tracking to nnUNet prediction results for instance segmentation
    
    Args:
        prediction: 3D numpy array with predicted binary segmentation
        axis: Tracking axis - 'xy', 'xz', or 'yz'
        iou_threshold: IoU threshold for matching instances across slices
        ioa_threshold: IoA threshold for matching instances across slices
        min_size: Minimum object size in voxels
        min_extent: Minimum extent in any dimension
        max_size: Maximum object size in voxels (None = no limit)
        remove_border_instances: Remove instances touching volume borders
        max_aspect_ratio: Maximum aspect ratio (None = no limit)
        
    Returns:
        instance_result: 3D numpy array with instance segmentation
    """
    print(f"Applying IoU tracking post-processing (axis={axis})...")
    
    # Convert prediction to binary mask (foreground > 0)
    binary_mask = (prediction == 1).astype(bool)
    
    # Apply IoU tracking
    instance_result = binary_volume_to_instances_iou_tracking(
        binary_mask,
        axis=axis,
        iou_threshold=iou_threshold,
        ioa_threshold=ioa_threshold,
        min_size=min_size,
        min_extent=min_extent,
        max_size=max_size,
        remove_border_instances=remove_border_instances,
        max_aspect_ratio=max_aspect_ratio,
        class_id=1,
        label_divisor=1000,
        verbose=True
    )
    
    print(f"IoU tracking completed. Found {len(np.unique(instance_result))-1} objects.")
    return instance_result.astype(np.uint16)


# ============================================================================
# Evaluation Functions (Integrated from evaluate_mito_pred.py)
# ============================================================================

def compute_precision_recall_f1(pred_mask, true_mask):
    """
    Compute precision, recall, and F1 score given prediction and ground truth masks.

    Parameters
    ----------
    pred_mask : ndarray
        Prediction mask.
    true_mask : ndarray
        Ground truth mask.

    Returns
    -------
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    """
    TP = np.sum((pred_mask == 1) & (true_mask == 1))
    FP = np.sum((pred_mask == 1) & (true_mask == 0))
    FN = np.sum((pred_mask == 0) & (true_mask == 1))

    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def instance_matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images."""
    # Check if the input arrays are valid
    _check_label_array(y_true, 'y_true')
    _check_label_array(y_pred, 'y_pred')

    y_true.shape == y_pred.shape or _raise(ValueError(
        "y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes".format(y_true=y_true,
                                                                                           y_pred=y_pred)))
    criterion in matching_criteria or _raise(ValueError("Matching criterion '%s' not supported." % criterion))

    if thresh is None:
        thresh = 0
    thresh = float(thresh) if np.isscalar(thresh) else map(float, thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)
    map_rev_true = np.array(map_rev_true)
    map_rev_pred = np.array(map_rev_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    assert 0 <= np.min(scores) <= np.max(scores) <= 1

    # Ignoring background
    scores = scores[1:, 1:]
    n_true, n_pred = scores.shape
    n_matched = min(n_true, n_pred)

    # Calculate true positives, false positives, and false negatives
    tp = (scores >= thresh).sum()
    fp = n_pred - tp
    fn = n_true - tp

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        'criterion': criterion,
        'thresh': thresh,
        'fp': fp,
        'tp': tp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0,
        'f1': f1,
        'n_true': n_true,
        'n_pred': n_pred,
        'mean_true_score': scores.mean() if n_true > 0 else 0,
        'mean_matched_score': scores[scores >= thresh].mean() if tp > 0 else 0,
        'panoptic_quality': tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0,
    }

    if report_matches:
        matched_pairs = [(i+1, np.argmax(scores[i])+1) for i in range(n_true)]
        matched_scores = [scores[i-1, j-1] for i, j in matched_pairs]
        matched_pairs = [(map_rev_true[i], map_rev_pred[j]) for i, j in matched_pairs]
        result.update({
            'matched_pairs': matched_pairs,
            'matched_scores': matched_scores,
        })

    return result


def evaluate_single_prediction(pred, gt_file, pred_file_name, save_dir=None):
    """
    Evaluate a single prediction against ground truth.
    
    Parameters
    ----------
    pred : ndarray
        Prediction array
    gt_file : str
        Path to the ground truth file
    pred_file_name : str
        Name of the prediction file (for saving results)
    save_dir : str
        Directory to save evaluation results
        
    Returns
    -------
    dict
        Dictionary containing metrics and file info
    """
    try:
        # Load ground truth
        y_true = load_image(gt_file)
        y_pred = pred
        
        # Calculate instance matching metrics
        metrics = instance_matching(y_true, y_pred, report_matches=True, thresh=0.5)
        
        # Calculate binary recall and precision
        binary_recall, binary_precision, binary_f1 = compute_precision_recall_f1(
            y_pred > 0, y_true > 0
        )
        metrics["binary_recall"] = binary_recall
        metrics["binary_precision"] = binary_precision
        metrics["binary_f1"] = binary_f1
        
        # Add file information
        metrics["pred_file"] = pred_file_name
        metrics["gt_file"] = gt_file
        metrics["file_name"] = pred_file_name
        
        # Save results if requested
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                
                base_name = Path(pred_file_name).stem
                if base_name.endswith('.nii'):
                    base_name = base_name[:-4]  # Remove .nii from .nii.gz
                
                txt_file = os.path.join(save_dir, f"{base_name}_metrics.txt")
                csv_file = os.path.join(save_dir, f"{base_name}_scores.csv")
                
                # Save metrics as txt file
                with open(txt_file, 'w') as file:
                    for key, value in metrics.items():
                        if key not in ["matched_pairs", "matched_scores", "pred_file", "gt_file", "file_name"]:
                            file.write(f'{key}: {value}\n')
                
                # Save matched pairs and scores as csv
                df_scores = pd.DataFrame({
                    'matched_pairs': [str(pair) for pair in metrics.get('matched_pairs', [])],
                    'matched_scores': metrics.get('matched_scores', [])
                })
                df_scores.to_csv(csv_file, index=False)
                
                print(f"  Evaluation results saved to: {save_dir}")
                
            except Exception as e:
                print(f"  Failed to save metrics for {pred_file_name}: {e}")
                
        return metrics
        
    except Exception as e:
        print(f"  Error evaluating {pred_file_name}: {e}")
        return {
            "pred_file": pred_file_name,
            "gt_file": gt_file,
            "file_name": pred_file_name,
            "error": str(e)
        }


def calculate_summary_statistics(results):
    """
    Calculate summary statistics across all evaluation results.
    
    Parameters
    ----------
    results : list
        List of evaluation result dictionaries
        
    Returns
    -------
    summary : dict
        Dictionary containing summary statistics
    """
    # Extract numeric metrics
    numeric_metrics = ['precision', 'recall', 'f1', 'accuracy', 'panoptic_quality',
                      'binary_precision', 'binary_recall', 'binary_f1',
                      'mean_true_score', 'mean_matched_score']
    
    summary = {}
    for metric in numeric_metrics:
        values = [r.get(metric, 0) for r in results if metric in r]
        if values:
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
    
    # Count statistics
    summary['total_files'] = len(results)
    summary['total_true_instances'] = sum(r.get('n_true', 0) for r in results)
    summary['total_pred_instances'] = sum(r.get('n_pred', 0) for r in results)
    summary['total_tp'] = sum(r.get('tp', 0) for r in results)
    summary['total_fp'] = sum(r.get('fp', 0) for r in results)
    summary['total_fn'] = sum(r.get('fn', 0) for r in results)
    
    # Overall metrics
    if summary['total_tp'] + summary['total_fp'] > 0:
        summary['overall_precision'] = summary['total_tp'] / (summary['total_tp'] + summary['total_fp'])
    else:
        summary['overall_precision'] = 0
        
    if summary['total_tp'] + summary['total_fn'] > 0:
        summary['overall_recall'] = summary['total_tp'] / (summary['total_tp'] + summary['total_fn'])
    else:
        summary['overall_recall'] = 0
        
    if summary['overall_precision'] + summary['overall_recall'] > 0:
        summary['overall_f1'] = 2 * (summary['overall_precision'] * summary['overall_recall']) / \
                               (summary['overall_precision'] + summary['overall_recall'])
    else:
        summary['overall_f1'] = 0
    
    return summary


def save_summary_results(results, summary, output_dir):
    """
    Save summary results to files.
    
    Parameters
    ----------
    results : list
        List of evaluation results
    summary : dict
        Summary statistics
    output_dir : str
        Directory to save results
    """
    try:
        # Create summary directory
        summary_dir = os.path.join(output_dir, "evaluation_summary")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Save detailed results as CSV
        results_df = pd.DataFrame(results)
        results_file = os.path.join(summary_dir, "detailed_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Save summary statistics
        summary_file = os.path.join(summary_dir, "summary_statistics.txt")
        with open(summary_file, 'w') as f:
            f.write("Evaluation Summary Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        # Save summary as CSV
        summary_df = pd.DataFrame([summary])
        summary_csv = os.path.join(summary_dir, "summary_statistics.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary Results saved to: {summary_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Failed to save summary results: {e}")


def find_matching_gt_file(input_file, gt_path):
    """
    Find the matching ground truth file for a given input file.
    
    Parameters
    ----------
    input_file : Path
        Path to the input file
    gt_path : str
        Path to ground truth file or directory
        
    Returns
    -------
    str or None
        Path to matching ground truth file, or None if not found
    """
    gt_path = Path(gt_path)
    
    # Get input file basename without _0000 suffix if present
    input_name = input_file.stem
    if input_file.suffix == '.gz':  # Handle .nii.gz
        input_name = input_file.with_suffix('').stem
    if input_name.endswith('_0000'):
        input_name = input_name[:-5]
    
    # If gt_path is a file, return it directly
    if gt_path.is_file():
        return str(gt_path)
    
    # If gt_path is a directory, search for matching file
    if gt_path.is_dir():
        # Try to find exact match with same extension
        file_ext = get_file_extension(str(input_file))
        possible_names = [
            f"{input_name}{file_ext}",
            f"{input_name}_0000{file_ext}",
            f"{input_name}.tif",
            f"{input_name}.tiff",
            f"{input_name}.nii.gz",
            f"{input_name}.nii",
        ]
        
        for name in possible_names:
            candidate = gt_path / name
            if candidate.exists():
                return str(candidate)
        
        # Search recursively
        for name in possible_names:
            candidates = list(gt_path.glob(f"**/{name}"))
            if candidates:
                return str(candidates[0])
    
    return None


def get_supported_files(input_path):
    """
    Get all supported image files from input path (file or directory)
    """
    input_path = Path(input_path)
    supported_extensions = ['.tif', '.tiff', '.nii.gz', '.nii']
    
    if input_path.is_file():
        # Single file
        if any(str(input_path).endswith(ext) for ext in supported_extensions):
            return [(input_path, get_file_extension(str(input_path)))]
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
    elif input_path.is_dir():
        # Directory - find all supported files
        files = []
        for ext in supported_extensions:
            pattern = f"**/*{ext}" if ext != '.nii.gz' else "**/*.nii.gz"
            found_files = list(input_path.glob(pattern))
            for file_path in found_files:
                files.append((file_path, get_file_extension(str(file_path))))
        if not files:
            raise ValueError(f"No supported image files found in directory: {input_path}")
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def get_file_extension(file_path):
    """
    Get the file extension, handling .nii.gz specially
    """
    if file_path.endswith('.nii.gz'):
        return '.nii.gz'
    else:
        return Path(file_path).suffix

def init_predictor(model_type='binary'):
    """
    Initialize nnUNet predictor with specified model type
    
    Args:
        model_type: Type of model to use ('binary' or 'binary_contour')
        
    Returns:
        predictor: Initialized nnUNet predictor
    """
    # Map model types to dataset paths
    model_paths = {
        'binary': 'Dataset014_mitolab/nnUNetTrainer__nnUNetPlans__2d',
        'binary_contour': 'Dataset010_mitolab_bc/nnUNetTrainer__nnUNetPlans__2d'
    }
    
    if model_type not in model_paths:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'binary' or 'binary_contour'")
    
    model_path = model_paths[model_type]
    print(f"Initializing predictor with model type: {model_type}")
    print(f"Model path: {model_path}")
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, model_path),
        use_folds=('all',),
        checkpoint_name='checkpoint_best.pth',
    )
    
    return predictor

def my_iterator(predictor, list_of_input_arrs, list_of_input_props):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
    for a, p in zip(list_of_input_arrs, list_of_input_props):
        data, seg = preprocessor.run_case_npy(a,
                                                None,
                                                p,
                                                predictor.plans_manager,
                                                predictor.configuration_manager,
                                                predictor.dataset_json)
        yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': p, 'ofile': None}

def pred_mito(predictor, img, save_name=None, input_format=None, 
              num_processes=3, batch_size=None, disable_parallel=False):
    """
    predict mitochondria from 2d nnUNet model on an image with optimized parallelization
    
    Args:
        predictor: nnUNet predictor
        img: input image array
        save_name: optional path to save prediction (if None, only returns prediction)
        input_format: file format for saving
        num_processes: number of processes for parallel segmentation export
        batch_size: batch size for processing slices (None for auto)
        disable_parallel: whether to disable parallel processing
        
    Returns:
        pred: 3D numpy array with prediction results
    """
    print(f"Processing image with shape: {img.shape}")
    
    # Create slice list with optimized batch processing
    slice_list = [img[np.newaxis,i:i+1, :, :] for i in range(img.shape[0])]
    properties_list = [{'spacing': [1.0, 1.0, 1.0]} for _ in range(len(slice_list))]
    
    # Determine optimal batch size if not specified
    if batch_size is None:
        # Auto-determine batch size based on image size and available memory
        if img.shape[0] <= 32:
            batch_size = img.shape[0]  # Process all slices at once for small volumes
        else:
            batch_size = min(16, img.shape[0])  # Process in smaller batches for large volumes
    
    print(f"Using batch size: {batch_size}, num_processes: {num_processes if not disable_parallel else 1}")
    
    # Process slices in batches for better memory management
    all_predictions = []
    
    for batch_start in range(0, len(slice_list), batch_size):
        batch_end = min(batch_start + batch_size, len(slice_list))
        batch_slices = slice_list[batch_start:batch_end]
        batch_properties = properties_list[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(slice_list)-1)//batch_size + 1}: "
              f"slices {batch_start}-{batch_end-1}")
        
        # Set number of processes
        processes = 1 if disable_parallel else num_processes
        
        # Predict batch
        batch_ret = predictor.predict_from_data_iterator(
            my_iterator(predictor, batch_slices, batch_properties),
            save_probabilities=False, 
            num_processes_segmentation_export=processes
        )
        
        all_predictions.extend(batch_ret)
    
    print(f"Prediction completed. Total slices processed: {len(all_predictions)}")
    
    # Convert ret to a 3d image
    pred = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for i in range(img.shape[0]):
        pred[i] = np.squeeze(all_predictions[i][0])
    
    # Save prediction if save_name is provided
    if save_name is not None and input_format is not None:
        save_prediction(pred, save_name, input_format)
    
    return pred
def main():
    parser = argparse.ArgumentParser(description="Predict mitochondria using 2D nnUNet model with IoU tracking post-processing and evaluation")
    parser.add_argument("-i", "--input", required=True, 
                       help="Input file or directory containing images")
    parser.add_argument("-o", "--output", required=True,
                       help="Output directory for predictions")
    parser.add_argument("--suffix", default="_pred",
                       help="Suffix to add to output filenames (default: _pred)")
    parser.add_argument("--no-iou-tracking", action="store_true",
                       help="Skip IoU tracking post-processing")
    parser.add_argument("--num-processes", type=int, default=3,
                       help="Number of processes for parallel segmentation export (default: 3)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for processing slices (default: auto)")
    parser.add_argument("--disable-parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--eval", action="store_true",
                       help="Enable evaluation against ground truth")
    parser.add_argument("--gt", type=str, default=None,
                       help="Path to ground truth file or directory (required if --eval is used)")
    
    # IoU tracking parameters
    parser.add_argument("--tracking-axis", type=str, default="xy", choices=['xy', 'xz', 'yz'],
                       help="Tracking axis for IoU tracking (default: xy)")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                       help="IoU threshold for matching instances across slices (default: 0.3)")
    parser.add_argument("--ioa-threshold", type=float, default=0.5,
                       help="IoA threshold for matching instances across slices (default: 0.5)")
    parser.add_argument("--min-size", type=int, default=500,
                       help="Minimum object size in voxels (default: 500)")
    parser.add_argument("--min-extent", type=int, default=3,
                       help="Minimum extent in any dimension (default: 3)")
    parser.add_argument("--max-size", type=int, default=None,
                       help="Maximum object size in voxels (default: None)")
    parser.add_argument("--remove-border-instances", action="store_true",
                       help="Remove instances touching the volume borders")
    parser.add_argument("--max-aspect-ratio", type=float, default=None,
                       help="Maximum aspect ratio (max_span/min_span, None = no limit)")
    
    # Model selection parameters
    parser.add_argument("--model-type", type=str, default="binary", 
                       choices=['binary', 'binary_contour'],
                       help="Type of model to use: 'binary' (Dataset014_mitolab) or 'binary_contour' (Dataset010_mitolab_bc) (default: binary)")
    
    args = parser.parse_args()
    
    # Validate evaluation parameters
    if args.eval and args.gt is None:
        parser.error("--gt must be specified when --eval is enabled")
    
    if args.eval:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            parser.error(f"Ground truth path does not exist: {args.gt}")
    
    # Get all supported files from input
    input_files = get_supported_files(args.input)
    print(f"Found {len(input_files)} files to process")
    
    # Create output directories
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for raw and iou_tracking results
    raw_output_path = output_path / "raw"
    iou_tracking_output_path = output_path / "iou_tracking"
    
    if not args.no_iou_tracking:
        raw_output_path.mkdir(exist_ok=True)
        iou_tracking_output_path.mkdir(exist_ok=True)
        print(f"Raw predictions will be saved to: {raw_output_path}")
        print(f"IoU tracking results will be saved to: {iou_tracking_output_path}")
    else:
        print(f"Predictions will be saved to: {output_path}")
    
    # Optimize parallel processing settings
    optimized_num_processes, optimized_batch_size = optimize_parallel_settings(
        args.num_processes, args.batch_size
    )
    
    # Initialize the predictor with specified model type
    predictor = init_predictor(model_type=args.model_type)

    # Initialize evaluation results storage
    evaluation_results = []
    
    # Process each file
    for i, (input_file, file_ext) in enumerate(input_files):
        print(f"\nProcessing {i+1}/{len(input_files)}: {input_file}")
        
        try:
            # Load the image
            img = load_image(input_file)
            
            # Generate input name for output files
            input_name = input_file.stem
            if input_file.suffix == '.gz':  # Handle .nii.gz
                input_name = input_file.with_suffix('').stem
            
            # Remove _0000 suffix if present
            if input_name.endswith('_0000'):
                input_name = input_name[:-5]  # Remove last 5 characters (_0000)
            
            # Predict mitochondria with optimized parallelization
            prediction = pred_mito(predictor, img, 
                                 num_processes=optimized_num_processes,
                                 batch_size=optimized_batch_size,
                                 disable_parallel=args.disable_parallel)
            
            if args.no_iou_tracking:
                # Save only raw prediction (no suffix)
                output_file = output_path / f"{input_name}{file_ext}"
                save_prediction(prediction, output_file, file_ext)
                
                # Evaluate raw prediction if requested
                if args.eval:
                    print(f"Evaluating raw prediction...")
                    gt_file = find_matching_gt_file(input_file, args.gt)
                    if gt_file:
                        eval_dir = output_path / "evaluation"
                        metrics = evaluate_single_prediction(
                            prediction, gt_file, f"{input_name}{file_ext}", 
                            save_dir=str(eval_dir)
                        )
                        if "error" not in metrics:
                            evaluation_results.append(metrics)
                            print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    else:
                        print(f"  Warning: No matching ground truth found for {input_file}")
            else:
                # Save raw prediction (no suffix)
                raw_output_file = raw_output_path / f"{input_name}{file_ext}"
                save_prediction(prediction, raw_output_file, file_ext)
                
                # Evaluate raw prediction if requested
                if args.eval:
                    print(f"Evaluating raw prediction...")
                    gt_file = find_matching_gt_file(input_file, args.gt)
                    if gt_file:
                        eval_dir = raw_output_path.parent / "raw_evaluation"
                        metrics = evaluate_single_prediction(
                            prediction, gt_file, f"{input_name}{file_ext}", 
                            save_dir=str(eval_dir)
                        )
                        if "error" not in metrics:
                            print(f"  Raw F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                
                # Apply IoU tracking and save
                iou_tracking_result = apply_iou_tracking(
                    prediction,
                    axis=args.tracking_axis,
                    iou_threshold=args.iou_threshold,
                    ioa_threshold=args.ioa_threshold,
                    min_size=args.min_size,
                    min_extent=args.min_extent,
                    max_size=args.max_size,
                    remove_border_instances=args.remove_border_instances,
                    max_aspect_ratio=args.max_aspect_ratio
                )
                iou_tracking_output_file = iou_tracking_output_path / f"{input_name}{file_ext}"
                save_prediction(iou_tracking_result, iou_tracking_output_file, file_ext)
                
                # Evaluate IoU tracking result if requested
                if args.eval:
                    print(f"Evaluating IoU tracking prediction...")
                    gt_file = find_matching_gt_file(input_file, args.gt)
                    if gt_file:
                        eval_dir = iou_tracking_output_path.parent / "iou_tracking_evaluation"
                        metrics = evaluate_single_prediction(
                            iou_tracking_result, gt_file, f"{input_name}{file_ext}", 
                            save_dir=str(eval_dir)
                        )
                        if "error" not in metrics:
                            evaluation_results.append(metrics)
                            print(f"  IoU Tracking F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    else:
                        print(f"  Warning: No matching ground truth found for {input_file}")
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue
    
    print(f"\nProcessing completed. Processed {len(input_files)} files.")
    if not args.no_iou_tracking:
        print(f"Raw predictions saved in: {raw_output_path}")
        print(f"IoU tracking results saved in: {iou_tracking_output_path}")
    
    # Calculate and save evaluation summary if evaluation was performed
    if args.eval and evaluation_results:
        print(f"\n{'='*60}")
        print("Calculating evaluation summary...")
        summary = calculate_summary_statistics(evaluation_results)
        
        # Print key metrics
        print(f"\nOverall Metrics (across {summary['total_files']} files):")
        print(f"  Overall F1 Score: {summary['overall_f1']:.4f}")
        print(f"  Overall Precision: {summary['overall_precision']:.4f}")
        print(f"  Overall Recall: {summary['overall_recall']:.4f}")
        print(f"  Mean F1 Score: {summary.get('f1_mean', 0):.4f} ± {summary.get('f1_std', 0):.4f}")
        print(f"  Mean Precision: {summary.get('precision_mean', 0):.4f} ± {summary.get('precision_std', 0):.4f}")
        print(f"  Mean Recall: {summary.get('recall_mean', 0):.4f} ± {summary.get('recall_std', 0):.4f}")
        
        # Save summary results
        save_summary_results(evaluation_results, summary, str(output_path))
if __name__ == '__main__':
    main()



