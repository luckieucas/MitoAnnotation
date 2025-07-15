import os
import argparse
import numpy as np
import torch
from scipy.ndimage import label
import tifffile as tiff
from skimage.morphology import binary_erosion
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

# It is assumed that you have added micro_sam to your Python path.
from micro_sam.util import get_sam_model, precompute_image_embeddings, set_precomputed, get_centers_and_bounding_boxes
from micro_sam.prompt_based_segmentation import segment_from_points
from micro_sam.prompt_generators import PointAndBoxPromptGenerator

def segment_slice_and_get_boundary(
    predictor,
    image_slice,
    labels_slice,
    n_points_per_label=1,
    erosion_width=1,
    label_list=None
):
    """
    Segments a single 2D slice and extracts the boundaries of the labeled regions.

    Args:
        predictor: The SAM predictor.
        image_slice: The input 2D image slice.
        labels_slice: The 2D label image with multiple regions.
        n_points_per_label (int): The number of random points to sample in each labeled region.
        erosion_width (int): The width of the erosion for boundary extraction.
        label_list (list, optional): A list of specific label IDs to process. If None, all labels are processed.

    Returns:
        np.ndarray: A 2D array containing the extracted boundaries.
    """
    labels = labels_slice.astype(int)

    # Ensure the image is 8-bit, as expected by the model's preprocessing.
    if image_slice.dtype != np.uint8:
        max_val = image_slice.max()
        if max_val > 0:
            image_slice = (image_slice / max_val * 255).astype(np.uint8)
        else:
            image_slice = image_slice.astype(np.uint8)

    # Precompute image embeddings for the current slice.
    image_embeddings = precompute_image_embeddings(predictor, image_slice)
    set_precomputed(predictor, image_embeddings)

    slice_boundaries_combined = np.zeros_like(labels, dtype=np.uint8)

    unique_labels_in_slice = np.unique(labels)
    unique_labels_in_slice = unique_labels_in_slice[unique_labels_in_slice != 0]

    # If a label_list is provided, filter the labels to process.
    if label_list:
        labels_to_process = [lbl for lbl in unique_labels_in_slice if lbl in label_list]
    else:
        labels_to_process = unique_labels_in_slice

    if not labels_to_process:
        return slice_boundaries_combined

    prompt_generator = PointAndBoxPromptGenerator(
        n_positive_points=n_points_per_label,
        n_negative_points=0,
        dilation_strength=10,
        get_box_prompts=False,
        get_point_prompts=True
    )

    _, bboxes = get_centers_and_bounding_boxes(labels, mode='p')

    for label_id in labels_to_process:
        binary_mask = (labels == label_id)
        current_bbox = bboxes.get(label_id)
        if current_bbox is None:
            continue

        binary_mask_tensor = torch.from_numpy(binary_mask[np.newaxis, np.newaxis, :, :]).float()
        points_coords, point_labels, _, _ = prompt_generator(binary_mask_tensor, [current_bbox])

        points_coords = points_coords.numpy().squeeze(axis=0)
        point_labels = point_labels.numpy().squeeze(axis=0)

        current_label_boundaries = []
        for i in range(len(points_coords)):
            single_point_coord = points_coords[i:i+1]
            single_point_label = point_labels[i:i+1]
            
            # Segment the mask from the generated point prompt.
            mask, _, _ = segment_from_points(predictor, single_point_coord[:, ::-1], single_point_label, return_all=True)
            mask = mask.squeeze()
            
            # Find the boundary by XORing the mask with its eroded version.
            eroded_mask = binary_erosion(mask, footprint=np.ones((erosion_width, erosion_width)))
            boundary = mask ^ eroded_mask
            current_label_boundaries.append(boundary)

        if current_label_boundaries:
            # Combine all boundaries found for the current label and add them to the slice.
            combined_boundary = np.logical_or.reduce(current_label_boundaries)
            slice_boundaries_combined[combined_boundary] = label_id
            
    return slice_boundaries_combined

def process_volume_for_one_dimension(
    image_volume,
    label_volume,
    predictor,
    axis,
    n_points_per_label,
    erosion_width,
    label_list=None
):
    """
    Processes a 3D volume along a single dimension (axis).

    Args:
        image_volume: The 3D image volume.
        label_volume: The 3D label volume.
        predictor: The SAM predictor.
        axis (int): The axis to process (0 for Z, 1 for Y, 2 for X).
        n_points_per_label (int): The number of points to sample per label.
        erosion_width (int): The width for the erosion operation.
        label_list (list, optional): A list of specific label IDs to process.

    Returns:
        np.ndarray: A 3D array of the boundaries for the processed dimension.
    """
    # Transpose the volumes so we can iterate over the chosen axis.
    if axis == 0:  # Z-axis slicing
        img_transposed = image_volume
        lbl_transposed = label_volume
    else:  # Y or X-axis slicing
        axes_order = [axis] + [d for d in [0, 1, 2] if d != axis]
        img_transposed = np.transpose(image_volume, axes_order)
        lbl_transposed = np.transpose(label_volume, axes_order)
    
    n_slices = img_transposed.shape[0]
    volume_boundaries_transposed = np.zeros_like(lbl_transposed, dtype=np.uint8)

    # Iterate over each slice in the transposed volume.
    for i in tqdm(range(n_slices), desc=f"Processing Axis {['Z', 'Y', 'X'][axis]}"):
        image_slice = img_transposed[i]
        labels_slice = lbl_transposed[i]

        if np.max(labels_slice) == 0:
            continue

        slice_boundaries = segment_slice_and_get_boundary(
            predictor,
            image_slice,
            labels_slice,
            n_points_per_label=n_points_per_label,
            erosion_width=erosion_width,
            label_list=label_list
        )
        volume_boundaries_transposed[i] = slice_boundaries
    
    # Transpose the result back to the original orientation if needed.
    if axis == 0:
        return volume_boundaries_transposed
    else:
        original_order = np.argsort(axes_order)
        return np.transpose(volume_boundaries_transposed, original_order)

def fill_holes_3d_slice_by_slice(mask_3d: np.ndarray) -> np.ndarray:
    """
    (Helper function)
    Performs hole-filling on every slice of a 3D **binary** mask along each dimension.
    It merges the results from all three orientations (XY, XZ, YZ planes).

    Args:
        mask_3d (np.ndarray): A 3D boolean or binary (0 and 1) array.

    Returns:
        np.ndarray: A new 3D boolean mask with holes filled from all slice orientations.
    """
    if mask_3d.ndim != 3:
        raise ValueError("Input to the helper function must be a 3D array.")
    filled_mask_combined = np.copy(mask_3d)
    # Iterate over the three axes (0, 1, 2)
    for axis in range(3):
        # Iterate over each slice along the current axis
        for i in range(mask_3d.shape[axis]):
            slice_2d = mask_3d.take(indices=i, axis=axis)
            filled_slice_2d = binary_fill_holes(slice_2d)
            # Use indexing to merge the filled slice back into the combined mask.
            if axis == 0:
                filled_mask_combined[i, :, :] = np.logical_or(filled_mask_combined[i, :, :], filled_slice_2d)
            elif axis == 1:
                filled_mask_combined[:, i, :] = np.logical_or(filled_mask_combined[:, i, :], filled_slice_2d)
            else:  # axis == 2
                filled_mask_combined[:, :, i] = np.logical_or(filled_mask_combined[:, :, i], filled_slice_2d)
    return filled_mask_combined

def process_multi_label_mask(multi_label_mask: np.ndarray, background_label: int = 0) -> np.ndarray:
    """
    Processes a multi-label 3D mask by performing slice-wise hole filling
    for each label independently.

    Args:
        multi_label_mask (np.ndarray): A 3D integer array representing the multi-label mask.
        background_label (int): The integer value representing the background. Defaults to 0.

    Returns:
        np.ndarray: A new multi-label 3D mask with holes inside each label filled.
    """
    filled_mask_final = np.copy(multi_label_mask)
    unique_labels = np.unique(multi_label_mask)
    labels_to_process = np.delete(unique_labels, np.where(unique_labels == background_label))
    if not labels_to_process.any():
        print("Warning: No foreground labels found in the mask. Returning the original mask.")
        return filled_mask_final
    print(f"Found labels to fill holes in: {labels_to_process}")
    # Iterate over each label that needs to be processed
    for label_val in labels_to_process:
        # 1. Create a temporary binary mask for the current label
        binary_mask_for_label = (multi_label_mask == label_val)
        # 2. Perform slice-by-slice hole filling on this binary mask
        filled_binary_mask = fill_holes_3d_slice_by_slice(binary_mask_for_label)
        # 3. Identify the newly filled regions that were originally background.
        hole_locations = filled_binary_mask & (multi_label_mask == background_label)
        # 4. In the final result mask, fill these hole locations with the current label's value
        filled_mask_final[hole_locations] = label_val
    return filled_mask_final

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Segment 3D volume and extract boundaries using SAM.")
    parser.add_argument('--img_path', type=str, required=True, help='Path to the input image TIFF file.')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the input mask TIFF file.')
    parser.add_argument('--model_type', type=str, default='vit_t', help='SAM model type (e.g., vit_t, vit_b, vit_l, vit_h).')
    parser.add_argument('--n_points', type=int, default=30, help='Number of points to sample per label for prompting.')
    parser.add_argument('--erosion_width', type=int, default=2, help='Width for the erosion operation to define boundary thickness.')
    parser.add_argument('--min_size', type=int, default=100, help='Minimum size (in voxels) for a connected component to be kept.')
    parser.add_argument('--label_list', type=int, nargs='*', help='Optional list of specific label IDs to process. If not provided, all labels are processed.')
    args = parser.parse_args()

    # --- Data Loading ---
    try:
        img_vol = tiff.imread(args.img_path)
        mask_vol = tiff.imread(args.mask_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the file paths.")
        exit()

    # --- Output Directory ---
    output_dir = os.path.dirname(args.img_path)
    print(f"All output files will be saved in: {output_dir}")

    # --- Model Initialization ---
    print(f"Initializing SAM model: {args.model_type}")
    predictor = get_sam_model(model_type=args.model_type)

    # --- Process All Three Dimensions ---
    all_boundaries = []
    for axis in range(3):
        axis_name = ['Z', 'Y', 'X'][axis]
        print(f"\n--- Starting processing for axis: {axis_name} ---")
        boundaries_for_axis = process_volume_for_one_dimension(
            img_vol, mask_vol, predictor, axis, args.n_points, args.erosion_width, args.label_list
        )
        
        save_path = os.path.join(output_dir, f"generated_width_{args.erosion_width}_boundaries_from_{axis_name}_slices.tiff")
        tiff.imwrite(save_path, boundaries_for_axis)
        print(f"Boundaries for axis {axis_name} saved to: {save_path}")
        
        all_boundaries.append(boundaries_for_axis)

    # --- Merge, Filter, and Post-process ---
    print("\n--- Merging boundaries from all dimensions ---")
    final_merged_boundaries = np.logical_or.reduce(all_boundaries).astype(np.uint8)
    
    # Filter the original mask by removing the found boundaries.
    mask_vol[final_merged_boundaries > 0] = 0
    print("Post-processing: Splitting disconnected components and filtering by size.")
    
    # Create a new mask to store the re-labeled results.
    processed_mask = np.zeros_like(mask_vol, dtype=np.uint16)
    current_new_label = 1
    
    # Get all unique labels except for the background (0).
    unique_labels = np.unique(mask_vol)
    unique_labels = unique_labels[unique_labels != 0]
    
    print(f"Found {len(unique_labels)} unique labels remaining to process.")
    
    for lbl in tqdm(unique_labels, desc="Splitting and re-labeling components"):
        # Create a binary mask for the current label.
        roi = (mask_vol == lbl)
        # Perform 3D connected component analysis.
        cc, num_features = label(roi)
        
        # Iterate through all found connected components.
        for comp_id in range(1, num_features + 1):
            comp_mask = (cc == comp_id)
            comp_size = np.sum(comp_mask)
            
            # Filter out small connected components.
            if comp_size >= args.min_size:
                # Assign a new, consecutive label to the valid component.
                processed_mask[comp_mask] = current_new_label
                current_new_label += 1
    
    print(f"Processing complete. Generated {current_new_label - 1} new labels.")
    
    # Save the component-filtered mask.
    tiff.imwrite(args.mask_path.replace(".tiff", "_cc3d.tiff"), processed_mask)
    
    # Apply hole-filling to the component-filtered mask and save it.
    print("Applying 3D hole filling...")
    processed_mask_filled = process_multi_label_mask(processed_mask)
    tiff.imwrite(args.mask_path.replace(".tiff", "_cc3d_fillholes.tiff"), processed_mask_filled)
    
    # Save the final merged boundaries mask based on the input mask's filename.
    merged_save_path = args.mask_path.replace(".tiff", "_boundaries_merged.tiff")
    tiff.imwrite(merged_save_path, final_merged_boundaries)
    print(f"Final merged boundaries saved to: {merged_save_path}")

if __name__ == '__main__':
    main()