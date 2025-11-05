import os
import argparse
import tifffile as tiff
import numpy as np
import torch
import json
import warnings
import sys
import glob
from pathlib import Path
from typing import Optional, Union, Tuple, List
from skimage.segmentation import relabel_sequential
from skimage import measure
import SimpleITK as sitk

# 假设您已经将 micro_sam 添加到了您的 Python 路径中
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.util import get_model_names

# Import nnUNet path utilities
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

# Import evaluation module
sys.path.append('/projects/weilab/liupeng/MitoAnnotation/src')
from evaluation.evaluate_res import evaluate_res, evaluate_directory

# --- 从 connectomics 库导入评估函数 (用于内部计算) ---
# 确保您的环境中已经安装了 connectomics 库
try:
    from connectomics.utils.evaluate import _check_label_array, _raise, matching_criteria, label_overlap
except ImportError:
    print("错误: 无法从 'connectomics.utils.evaluate' 导入评估函数。")
    print("请确保 'connectomics-pytorch' 已经正确安装在您的 Python 环境中。")
    print("您可以通过 'pip install connectomics-pytorch' 来安装它。")
    exit()

# --- 依赖于导入函数的评估函数 ---
def instance_matching(y_true, y_pred, thresh=0.5, criterion='iou', report_matches=False):
    """Calculate detection/instance segmentation metrics between ground truth and predicted label images."""
    # Check if the input arrays are valid
    _check_label_array(y_true, 'y_true')
    _check_label_array(y_pred, 'y_pred')

    if y_true.shape != y_pred.shape:
        _raise(ValueError(f"y_true ({y_true.shape}) and y_pred ({y_pred.shape}) have different shapes"))
    if criterion not in matching_criteria:
        _raise(ValueError(f"Matching criterion '{criterion}' not supported."))

    thresh = float(thresh)

    y_true, _, map_rev_true = relabel_sequential(y_true)
    y_pred, _, map_rev_pred = relabel_sequential(y_pred)
    map_rev_true = np.array(map_rev_true)
    map_rev_pred = np.array(map_rev_pred)

    overlap = label_overlap(y_true, y_pred, check=False)
    scores = matching_criteria[criterion](overlap)
    
    # Handle empty scores array
    if scores.size == 0:
        n_true = np.max(y_true) if y_true.size > 0 else 0
        n_pred = np.max(y_pred) if y_pred.size > 0 else 0
        tp, fp, fn = 0, n_pred, n_true
    else:
        assert 0 <= np.min(scores) <= np.max(scores) <= 1
        scores = scores[1:, 1:]
        n_true, n_pred = scores.shape
        # This is a many-to-one matching, not one-to-one, so we find matches from GT to Pred
        matches = scores >= thresh
        tp = np.count_nonzero(np.sum(matches, axis=1) > 0) # Count GT objects that have at least one match
        fp = n_pred - np.count_nonzero(np.sum(matches, axis=0) > 0) # Count Pred objects that have no match
        fn = n_true - tp # Count GT objects that have no match


    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    result = {
        'criterion': criterion,
        'thresh': thresh,
        'fp': int(fp),
        'tp': int(tp),
        'fn': int(fn),
        'precision': round(float(precision), 4), 
        'recall': round(float(recall), 4),
        'accuracy': round(float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0, 4),
        'f1': round(float(f1), 4),
        'n_true': int(n_true),
        'n_pred': int(n_pred),
        'mean_true_score': float(np.mean(np.max(scores, axis=1))) if n_true > 0 and n_pred > 0 else 0.0,
        'mean_matched_score': float(np.mean(scores[scores >= thresh])) if tp > 0 else 0.0,
        'panoptic_quality': float(tp / (tp + 0.5 * fp + 0.5 * fn)) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0,
    }

    if report_matches:
        if n_true > 0 and n_pred > 0:
            matched_pairs_indices = [(i, np.argmax(scores[i])) for i in range(n_true) if np.max(scores[i]) >= thresh]
            matched_pairs = [(map_rev_true[i+1], map_rev_pred[j+1]) for i, j in matched_pairs_indices]
            matched_scores = [scores[i, j] for i, j in matched_pairs_indices]
        else:
            matched_pairs, matched_scores = [], []
            
        result.update({
            'matched_pairs': matched_pairs,
            'matched_scores': matched_scores,
        })

    return result

# --- End of Evaluation Function ---

def postprocess_segmentation(segmentation: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Post-process segmentation by filtering out small instances and relabeling.
    
    Args:
        segmentation: Input segmentation mask with instance labels
        min_size: Minimum size (voxel count) for an instance to be kept
        
    Returns:
        Processed segmentation with small instances removed and consecutive labels
    """
    print(f"\nPost-processing segmentation...")
    print(f"  Original number of instances (including background): {len(np.unique(segmentation))}")
    
    # Filter out labels with size smaller than min_size
    print(f"  Filtering instances smaller than {min_size} voxels...")
    filtered_mask = segmentation.copy()
    unique_labels, counts = np.unique(filtered_mask, return_counts=True)
    small_labels = unique_labels[(counts < min_size) & (unique_labels != 0)]
    
    if small_labels.size > 0:
        print(f"  Removing {len(small_labels)} small instances.")
        filtered_mask[np.isin(filtered_mask, small_labels)] = 0
    else:
        print("  No small instances to remove.")
    
    # Relabel all remaining objects to be consecutive from 1
    print("  Relabeling to ensure consecutive numbering...")
    relabeled_mask, num_labels = measure.label(filtered_mask, background=0, return_num=True)
    print(f"  Final number of instances (excluding background): {num_labels}")
    
    return relabeled_mask

def run_automatic_instance_segmentation(
    image: np.ndarray,
    model_type: str = "vit_b_lm",
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    embedding_path: str = None
):
    print(f"Running Automatic Instance Segmentation with model '{model_type}'...")
    ndim = image.ndim
    if ndim not in [2, 3]:
        raise ValueError(f"Input image must be 2D or 3D, but has {ndim} dimensions")
        
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,
        device=device,
        amg=False,
        is_tiled=(tile_shape is not None),
    )

    prediction = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        embedding_path=embedding_path,
        ndim=ndim,
        tile_shape=tile_shape,
        halo=halo,
        verbose=True
    )
    return prediction

def get_nnunet_input_files(dataset_path: str) -> List[str]:
    """
    Get all image files (TIFF and NIfTI) from the imagesTs directory of an nnUNet dataset.
    
    Args:
        dataset_path: Path to the nnUNet dataset root directory
        
    Returns:
        List of paths to image files
    """
    images_dir = os.path.join(dataset_path, "imagesTs")
    if not os.path.exists(images_dir):
        raise ValueError(f"imagesTs directory not found: {images_dir}")
    
    # Find all TIFF and NIfTI files
    image_files = []
    # TIFF files
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    # NIfTI files
    for ext in ['*.nii.gz', '*.nii', '*.NII.GZ', '*.NII']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if not image_files:
        raise ValueError(f"No image files (TIFF or NIfTI) found in {images_dir}")
    
    image_files.sort()
    return image_files

def get_output_filename(input_filename: str) -> str:
    """
    Convert nnUNet input filename to output filename by removing _0000 suffix.
    Preserves the original file extension.
    
    Args:
        input_filename: Original filename (e.g., 'sample_0000.tiff' or 'sample_0000.nii.gz')
        
    Returns:
        Output filename (e.g., 'sample.tiff' or 'sample.nii.gz')
    """
    name = os.path.basename(input_filename)
    # Remove _0000 suffix if present
    if '_0000' in name:
        name = name.replace('_0000', '')
    return name

def is_nifti_file(filepath: str) -> bool:
    """
    Check if a file is a NIfTI file based on its extension.
    
    Args:
        filepath: Path to the file
        
    Returns:
        True if the file is a NIfTI file, False otherwise
    """
    filepath_lower = filepath.lower()
    return filepath_lower.endswith('.nii.gz') or filepath_lower.endswith('.nii')

def load_image(filepath: str) -> Tuple[np.ndarray, Optional[sitk.Image]]:
    """
    Load an image file (TIFF or NIfTI) and return the numpy array and optionally the SimpleITK image.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (numpy_array, sitk_image). sitk_image is None for TIFF files.
    """
    if is_nifti_file(filepath):
        print(f"Loading NIfTI file using SimpleITK...")
        sitk_image = sitk.ReadImage(filepath)
        numpy_array = sitk.GetArrayFromImage(sitk_image)
        return numpy_array, sitk_image
    else:
        print(f"Loading TIFF file using tifffile...")
        numpy_array = tiff.imread(filepath)
        return numpy_array, None

def save_image(filepath: str, image_array: np.ndarray, reference_image: Optional[sitk.Image] = None):
    """
    Save an image array to file (TIFF or NIfTI format).
    
    Args:
        filepath: Path where the image should be saved
        image_array: NumPy array containing the image data
        reference_image: Optional SimpleITK image to use as reference for NIfTI files (for metadata)
    """
    if is_nifti_file(filepath):
        print(f"Saving NIfTI file using SimpleITK...")
        # Convert to uint16 if needed
        if image_array.dtype != np.uint16:
            image_array = image_array.astype(np.uint16)
        
        # Create SimpleITK image from numpy array
        sitk_image = sitk.GetImageFromArray(image_array)
        
        # If reference image is provided, copy metadata (spacing, origin, direction)
        if reference_image is not None:
            sitk_image.SetSpacing(reference_image.GetSpacing())
            sitk_image.SetOrigin(reference_image.GetOrigin())
            sitk_image.SetDirection(reference_image.GetDirection())
        
        # Write the image
        sitk.WriteImage(sitk_image, filepath, useCompression=True)
        print("NIfTI file saved successfully with compression!")
    else:
        print(f"Saving TIFF file using tifffile...")
        tiff.imwrite(
            filepath,
            image_array.astype(np.uint16),
            compression="zlib"
        )
        print("TIFF file saved successfully with compression!")

def main():
    available_models = ", ".join(get_model_names())
    parser = argparse.ArgumentParser(
        description="Perform 3D Automatic Instance Segmentation on nnUNet dataset using a µsam model."
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, 
                        help="Dataset name or ID (e.g., Dataset001_MitoHardBeta or 1).")
    parser.add_argument("-l", "--label_path", default=None, help="(Optional) Path to a ground truth file (TIFF or NIfTI). If provided, it will be used to filter the prediction.")
    parser.add_argument("--mode", default="FT", help="Mode to use. Defaults to 'FT'. Available modes: FT, ZS. FT: Fine-tuning, ZS: Zero-shot.")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode (will use instancesTs as ground truth).")
    parser.add_argument("-m", "--model_type", default="vit_b_em_organelles", help=f"The SAM model type to use. Defaults to 'vit_b_em_organelles'. Available models: {available_models}")
    parser.add_argument("--use_embeddings", action="store_true", help="Enable caching of image embeddings (saves .zarr files in output directory).")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="(Optional) Path to a custom model checkpoint file.")
    parser.add_argument("--min_size", type=int, default=500, help="Minimum size (voxel count) for an instance to be kept. Default is 500.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=[512, 512], help="Tile shape for processing. Default is 512 512.")
    parser.add_argument("--halo", type=int, nargs=2, default=[16, 16], help="Halo size for tiles. Default is 16 16.")

    args = parser.parse_args()

    # Convert dataset name or ID to standard dataset name
    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    
    # Build dataset root directory
    dataset_root = os.path.join(nnUNet_raw, dataset_name)
    if not os.path.exists(dataset_root):
        raise ValueError(f"Dataset directory does not exist: {dataset_root}")
    
    print(f"nnUNet_raw: {nnUNet_raw}")
    print(f"Dataset input: {args.dataset}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset root: {dataset_root}")

    # Get all input files from the dataset
    print(f"\nScanning dataset: {dataset_root}")
    try:
        input_files = get_nnunet_input_files(dataset_root)
        print(f"Found {len(input_files)} file(s) to process")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create output directory
    output_dir = os.path.join(dataset_root, f"imagesTs_microsam_{args.mode}_pred")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each file
    output_files = []
    for idx, input_file in enumerate(input_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {idx}/{len(input_files)}: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        
        # Load image
        print(f"Loading image from {input_file}...")
        try:
            image, reference_image = load_image(input_file)
            print(f"Image loaded with shape: {image.shape}")
        except FileNotFoundError:
            print(f"Error: File not found at {input_file}")
            continue
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Generate embedding path if enabled
        embedding_path = None
        if args.use_embeddings:
            # Create embedding filename based on input filename
            base_name = os.path.splitext(get_output_filename(input_file))[0]
            embedding_path = os.path.join(output_dir, f"{base_name}_embeddings.zarr")
            print(f"Embedding cache path: {embedding_path}")
        
        # Run segmentation
        try:
            segmentation_result = run_automatic_instance_segmentation(
                image=image,
                model_type=args.model_type,
                checkpoint_path=args.checkpoint_path,
                tile_shape=args.tile_shape,
                halo=args.halo,
                embedding_path=embedding_path
            )
        except Exception as e:
            print(f"Error during segmentation: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # If a label_path is provided, use it for filtering
        if args.label_path:
            print(f"\nLoading label mask from: {args.label_path}")
            try:
                label_mask, _ = load_image(args.label_path)
                if label_mask.shape != segmentation_result.shape:
                     print(f"Warning: Shape mismatch between prediction ({segmentation_result.shape}) and label mask ({label_mask.shape}). Skipping filtering.")
                else:
                    # Filter the prediction using the label mask
                    print("Filtering segmentation result using the provided labels...")
                    segmentation_result[label_mask == 0] = 0
                    print("Filtering applied successfully.")

            except FileNotFoundError:
                print(f"Error: Label file not found at {args.label_path}. Skipping filtering.")
            except Exception as e:
                print(f"Error loading label file: {e}")
                import traceback
                traceback.print_exc()

        # Post-process the segmentation (filter small instances and relabel)
        segmentation_result = postprocess_segmentation(segmentation_result, min_size=args.min_size)

        # Generate output filename (remove _0000 suffix)
        output_filename = get_output_filename(input_file)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the segmentation result
        print(f"\nSaving segmentation to: {output_path} ...")
        try:
            # Use reference_image for NIfTI files to preserve metadata
            save_image(output_path, segmentation_result, reference_image=reference_image)
            output_files.append(output_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print(f"All segmentation tasks completed! Processed {len(output_files)}/{len(input_files)} files.")
    print("="*60)
    
    # If evaluation is enabled, evaluate against instancesTs
    if args.eval:
        gt_dir = os.path.join(dataset_root, "instancesTs")
        mask_dir = os.path.join(dataset_root, "masksTs")
        if not os.path.exists(gt_dir):
            print(f"\nWarning: Ground truth directory not found: {gt_dir}")
            print("Skipping evaluation.")
        else:
            print("\n" + "="*60)
            print("Starting evaluation...")
            print("="*60)
            print(f"  Prediction Directory: {output_dir}")
            print(f"  Ground Truth Directory: {gt_dir}")
            
            try:
                results, summary = evaluate_directory(
                    pred_dir=output_dir,
                    gt_dir=gt_dir,
                    mask_dir=mask_dir,
                    save_results=True
                )
                
                print("\n" + "-"*60)
                print("Evaluation Summary:")
                print("-"*60)
                for key, value in summary.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                        
            except Exception as e:
                print(f"\nError during evaluation: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)

if __name__ == '__main__':
    main()
