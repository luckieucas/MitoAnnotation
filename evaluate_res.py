
import argparse
import tifffile
import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential
from connectomics.utils.evaluate import _check_label_array,_raise,matching_criteria,label_overlap


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

def evaluate_res(pred_file="res_tif/pred_mask.tif", 
                 gt_file = '/mmfs1/data/liupen/project/dataset/mito/3dem_labeled/label/hela_cell_mito.tif'
                 ):
    """
    Evaluate the prediction result using instance matching metrics.

    Parameters
    ----------
    pred_file : str or np.ndarray
        The path to the prediction result tif file.
    gt_file : str or np.ndarray
        The path to the ground truth tif file.
    Returns
    -------
    metrics : dict
        A dictionary containing the instance matching metrics.
    """
    if isinstance(gt_file, str):
        y_true = tifffile.imread(gt_file)
    else:
        y_true = gt_file
    if isinstance(pred_file, str):
        y_pred = tifffile.imread(pred_file)
    else:
        y_pred = pred_file
    # Calculate instance matching metrics
    metrics = instance_matching(y_true, y_pred, report_matches=True)
    
    # Calculate binary recall and precision
    binary_recall, binary_precision, binary_f1 = compute_precision_recall_f1(
        y_pred>0, y_true>0
    )
    metrics["binary_recall"] = binary_recall
    metrics["binary_precision"] = binary_precision
    metrics["binary_f1"] = binary_f1
    # Save as txt file
    try:
        # Replace .tif or .tiff with .txt
        with open(pred_file.replace('.tiff','.txt').replace('.tif','.txt'), 'w') as file:
            for key, value in metrics.items():
                if key!="matched_pairs" and key!="matched_scores":
                    print(f"{key}: {value}")
                file.write(f'{key}: {value}\n')
        df = pd.DataFrame(metrics)
        df_scores = df[["matched_pairs", "matched_scores"]]
        df_scores.to_csv(pred_file.replace('.tiff','.csv').replace('.tif','.csv'))
    except:
        print(f"Failed to save metrics to txt")
        print(metrics)
    return metrics


if __name__ == '__main__':
    # Load the HDF5 datasets
    args = argparse.ArgumentParser()
    args.add_argument('--gt_file', type=str, default="hela_cell_em")
    args.add_argument('--pred_file', type=str, default="res_tif/pred_mask.tif")
    args = args.parse_args()
    print(f"evaluate pred_file: {args.pred_file} and gt_file: {args.gt_file}")
    # Calculate instance matching metrics
    metrics = evaluate_res(pred_file=args.pred_file, gt_file=args.gt_file)