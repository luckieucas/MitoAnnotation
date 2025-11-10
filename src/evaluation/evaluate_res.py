
import argparse
import tifffile
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from skimage.segmentation import relabel_sequential
from connectomics.utils.evaluate import _check_label_array,_raise,matching_criteria,label_overlap


def read_image_file(file_path):
    """
    Read image file using appropriate library based on file extension.
    Supports both TIFF (using tifffile) and NII.GZ (using SimpleITK) formats.
    
    Parameters
    ----------
    file_path : str
        Path to the image file
        
    Returns
    -------
    numpy.ndarray
        Image array
    """
    if isinstance(file_path, str):
        file_ext = os.path.splitext(file_path.lower())[-1]
        # Handle .nii.gz extension
        if file_path.lower().endswith('.nii.gz'):
            file_ext = '.nii.gz'
        
        if file_ext in ['.nii', '.nii.gz']:
            # Use SimpleITK for NII.GZ files
            sitk_image = sitk.ReadImage(file_path)
            return sitk.GetArrayFromImage(sitk_image)
        else:
            # Use tifffile for TIFF files
            return tifffile.imread(file_path)
    else:
        # If file_path is already an array, return it
        return file_path


def compute_precision_recall_f1(pred_mask, true_mask):
    # True Positive (TP), False Positive (FP), False Negative (FN)
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

    # Precision, Recall, F1, Accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    return precision, recall, f1, accuracy

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
        # matched_pairs = [(i, j) for i in range(n_true) for j in range(n_pred) if scores[i, j] >= thresh]
        # matched_scores = [scores[i, j] for i, j in matched_pairs]
        matched_pairs = [(i+1, np.argmax(scores[i])+1) for i in range(n_true)]
        matched_scores = [scores[i-1, j-1] for i, j in matched_pairs]
        matched_pairs = [(map_rev_true[i], map_rev_pred[j]) for i, j in matched_pairs]
        result.update({
            'matched_pairs': matched_pairs,
            'matched_scores': matched_scores,
        })

    return result

def evaluate_single_file(pred_file, gt_file, save_results=True, mask_file=None):
    """
    Evaluate a single prediction file against ground truth.
    Supports both TIFF and NII.GZ file formats.
    
    Parameters
    ----------
    pred_file : str or np.ndarray
        Path to the prediction file (TIFF or NII.GZ) or numpy array
    gt_file : str or np.ndarray
        Path to the ground truth file (TIFF or NII.GZ) or numpy array
    save_results : bool
        Whether to save results to files
    mask_file : str or np.ndarray, optional
        Path to the mask file (TIFF or NII.GZ) or numpy array. If provided, only evaluate regions where mask > 0
        
    Returns
    -------
    dict
        Dictionary containing metrics and file info
    """
    try:
        # Load images
        y_true = read_image_file(gt_file)
        y_pred = read_image_file(pred_file)
            
        # Apply mask if provided
        if mask_file is not None:
            mask = read_image_file(mask_file)
            
            # Check if mask shape matches
            if mask.shape != y_pred.shape:
                raise ValueError(f"Mask shape {mask.shape} does not match prediction shape {y_pred.shape}")
            
            # Apply mask: set regions where mask <= 0 to background (0)
            mask_bool = mask > 0
            y_pred = y_pred * mask_bool
            y_true = y_true * mask_bool
            
        # Calculate instance matching metrics
        metrics = instance_matching(y_true, y_pred, report_matches=True, thresh=0.5)
        
        # Calculate binary recall and precision
        binary_recall, binary_precision, binary_f1, bianry_accuracy = compute_precision_recall_f1(
            y_pred>0, y_true>0
        )
        metrics["binary_recall"] = binary_recall
        metrics["binary_precision"] = binary_precision
        metrics["binary_f1"] = binary_f1
        metrics["binary_accuracy"] = bianry_accuracy        
        # Add file information
        metrics["pred_file"] = pred_file
        metrics["gt_file"] = gt_file
        metrics["file_name"] = os.path.basename(pred_file)
        
        # Save results if requested
        if save_results and isinstance(pred_file, str):
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(pred_file) + "_evaluation"
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = os.path.splitext(os.path.basename(pred_file))[0]
                txt_file = os.path.join(output_dir, f"{base_name}_metrics.txt")
                csv_file = os.path.join(output_dir, f"{base_name}_scores.csv")
                
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
                
            except Exception as e:
                print(f"Failed to save metrics for {pred_file}: {e}")
                
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {pred_file}: {e}")
        return {
            "pred_file": pred_file,
            "gt_file": gt_file,
            "file_name": os.path.basename(pred_file),
            "error": str(e)
        }

def evaluate_res(pred_file="res_tif/pred_mask.tif", 
                 gt_file='/mmfs1/data/liupen/project/dataset/mito/3dem_labeled/label/hela_cell_mito.tif',
                 save_results=True,
                 mask_file=None):
    """
    Evaluate the prediction result using instance matching metrics.
    Supports both TIFF and NII.GZ file formats.
    
    Parameters
    ----------
    pred_file : str or np.ndarray
        The path to the prediction result file (TIFF or NII.GZ) or directory containing image files.
    gt_file : str or np.ndarray
        The path to the ground truth file (TIFF or NII.GZ) or directory containing image files.
    save_results : bool
        Whether to save results to files.
    mask_file : str, optional
        Path to the mask file. If provided, only evaluate regions where mask > 0.
        
    Returns
    -------
    metrics : dict or list
        A dictionary containing the instance matching metrics for single file,
        or a list of dictionaries for multiple files.
    """
    # Handle single file evaluation
    if os.path.isfile(pred_file) and os.path.isfile(gt_file):
        return evaluate_single_file(pred_file, gt_file, save_results, mask_file)
    
    # Handle directory evaluation
    elif os.path.isdir(pred_file) or os.path.isdir(gt_file):
        return evaluate_directory(pred_file, gt_file, mask_dir=mask_file, save_results=save_results, max_workers=max_workers)
    
    else:
        raise ValueError(f"Invalid input: {pred_file} or {gt_file}")

def evaluate_directory(pred_dir, gt_dir, mask_dir=None, save_results=True, max_workers=None):
    """
    Evaluate all image files (TIFF or NII.GZ) in prediction directory against corresponding files in ground truth directory.
    
    Parameters
    ----------
    pred_dir : str
        Path to directory containing prediction files (TIFF or NII.GZ)
    gt_dir : str
        Path to directory containing ground truth files (TIFF or NII.GZ)
    save_results : bool
        Whether to save individual results
    max_workers : int
        Maximum number of parallel workers. If None, uses CPU count.
    mask_file : str, optional
        Path to the mask file. If provided, only evaluate regions where mask > 0.
        
    Returns
    -------
    results : list
        List of dictionaries containing metrics for each file
    summary : dict
        Summary statistics across all files
    """
    # Find all image files in prediction directory (support both TIFF and NII.GZ)
    pred_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.nii.gz', '*.nii', '*.NII.GZ', '*.NII']
    pred_files = []
    for pattern in pred_patterns:
        pred_files.extend(glob.glob(os.path.join(pred_dir, pattern)))
    
    if not pred_files:
        raise ValueError(f"No image files (TIFF or NII.GZ) found in {pred_dir}")
    
    # Find corresponding ground truth files
    file_pairs = []
    for pred_file in pred_files:
        pred_name = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, pred_name)
        if mask_dir is not None:
            mask_file = os.path.join(mask_dir, pred_name)
            if not os.path.exists(mask_file):
                print(f"Warning: Mask file not found for {pred_file}")
                mask_file = None
        else:
            mask_file = None
        
        if os.path.exists(gt_file):
            file_pairs.append((pred_file, gt_file, mask_file))
        else:
            print(f"Warning: Ground truth file not found for {pred_file}")
    
    if not file_pairs:
        raise ValueError(f"No matching ground truth files found in {gt_dir}")
    
    print(f"Found {len(file_pairs)} file pairs to evaluate")
    
    # Set up parallel processing
    if max_workers is None:
        max_workers = min(cpu_count(), len(file_pairs))
    
    results = []
    failed_files = []
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pair = {
            executor.submit(evaluate_single_file, pred_file, gt_file, save_results, mask_file): (pred_file, gt_file)
            for pred_file, gt_file, mask_file in file_pairs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_pair):
            pred_file, gt_file = future_to_pair[future]
            try:
                result = future.result()
                if "error" in result:
                    failed_files.append((pred_file, result["error"]))
                else:
                    results.append(result)
                print(f"Completed: {os.path.basename(pred_file)}")
            except Exception as e:
                failed_files.append((pred_file, str(e)))
                print(f"Failed: {os.path.basename(pred_file)} - {e}")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Successfully processed: {len(results)} files")
    print(f"Failed: {len(failed_files)} files")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files:
            print(f"  {os.path.basename(file_path)}: {error}")
    
    # Calculate summary statistics
    if results:
        summary = calculate_summary_statistics(results)
        
        # Save summary
        if save_results:
            save_summary_results(results, summary, pred_dir)
        
        return results, summary
    else:
        print("No successful evaluations to summarize")
        return results, {}

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
                      'binary_precision', 'binary_recall', 'binary_f1', 'binary_accuracy',
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
        summary_dir = os.path.join(output_dir, "summary")
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
        
        print(f"\nSummary results saved to: {summary_dir}")
        
    except Exception as e:
        print(f"Failed to save summary results: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate prediction results')
    parser.add_argument('--gt_file', type=str, default="hela_cell_em",
                       help='Ground truth file or directory path')
    parser.add_argument('--pred_file', type=str, default="res_tif/pred_mask.tif",
                       help='Prediction file or directory path')
    parser.add_argument('--mask_file', type=str, default=None,
                       help='Mask file path. If provided, only evaluate regions where mask > 0')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: CPU count)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save individual results to files')
    
    args = parser.parse_args()
    
    save_results = not args.no_save
    max_workers = args.max_workers
    
    print(f"Evaluating pred_file: {args.pred_file} and gt_file: {args.gt_file}")
    if args.mask_file:
        print(f"Mask file: {args.mask_file}")
    print(f"Save results: {save_results}")
    if max_workers:
        print(f"Max workers: {max_workers}")
    
    # Evaluate
    if os.path.isdir(args.pred_file) or os.path.isdir(args.gt_file):
        results, summary = evaluate_directory(args.pred_file, args.gt_file, 
                                            save_results=save_results, max_workers=max_workers,
                                            mask_dir=args.mask_file)
        
        print("\nOverall Summary:")
        for key, value in summary.items():
            if key.endswith('_mean'):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        # Single file evaluation
        metrics = evaluate_res(pred_file=args.pred_file, gt_file=args.gt_file, 
                             save_results=save_results, mask_file=args.mask_file)
        print("\nMetrics:")
        for key, value in metrics.items():
            if key not in ["matched_pairs", "matched_scores"]:
                print(f"{key}: {value}")