import numpy as np
import tifffile as tiff
from scipy.ndimage import label
import argparse
from tqdm import tqdm

def process_mask(input_path, output_path, target_label, min_size=30):
    """
    Processes a 3D TIFF mask file using connected component analysis and filters out small components.

    - If target_label is 0, all labels are processed. Each connected component larger than 
      min_size will become a new object, and all labels in the final output mask will be 
      re-numbered to be consecutive, starting from 1.

    - If target_label is not 0, only the specified label is processed. Components smaller 
      than min_size are removed, while larger components are split into new, separate labels. 
      All other labels in the mask retain their original numbers. The new labels for the split 
      components will start from the original mask's maximum label + 1.

    Args:
        input_path (str): Path to the input 3D TIFF mask file.
        output_path (str): Path to save the processed mask file.
        target_label (int): The target label to process. Use 0 to process all labels.
        min_size (int): Minimum size (in voxels) for connected components to be kept.
    """
    # Read the input mask
    print(f"Reading mask file: {input_path}")
    mask = tiff.imread(input_path)
    
    if target_label == 0:
        # Mode 1: Process all labels and ensure the final labels are consecutive starting from 1.
        print("Processing all labels. The output mask's labels will be re-numbered to be consecutive.")
        
        # Create a new mask to store the results
        processed_mask = np.zeros_like(mask, dtype=np.uint16)
        current_new_label = 1
        
        # Get all unique labels except for the background (0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels != 0]
        
        print(f"Detected {len(unique_labels)} unique labels: {unique_labels}")
        
        for lbl in tqdm(unique_labels, desc="Splitting all labels"):
            # Create a binary mask for the current label
            roi = (mask == lbl)
            # Perform connected component analysis
            cc, num_features = label(roi)
            
            # Iterate through all found connected components
            for comp_id in range(1, num_features + 1):
                comp_mask = (cc == comp_id)
                comp_size = np.sum(comp_mask)
                
                # Filter out small connected components
                if comp_size >= min_size:
                    # Assign a new, consecutive label to the valid component
                    processed_mask[comp_mask] = current_new_label
                    current_new_label += 1
        
        print(f"Processing complete. Generated {current_new_label - 1} new labels.")

    else:
        # Mode 2: Process only the specified label and preserve all other labels.
        print(f"Processing only label {target_label}. Other labels will be preserved.")
        
        # Start processing from a copy of the original mask
        processed_mask = mask.copy()
        roi = (mask == target_label)
        
        # Check if the target label exists
        if not np.any(roi):
            print(f"Label {target_label} not found in the mask. No changes were made.")
        else:
            # Perform connected component analysis on the target label's region
            cc, num_features = label(roi)
            print(f"Found {num_features} connected components for label {target_label}.")
            
            # If there is only one component, no splitting is needed, just check its size
            if num_features <= 1:
                if np.sum(roi) < min_size:
                    print(f"The single component of label {target_label} is smaller than the minimum size and will be removed.")
                    processed_mask[roi] = 0
                else:
                    print(f"The single component of label {target_label} is large enough. No changes needed.")
            else:
                # If there are multiple components, proceed with splitting
                # The starting number for new labels will be the current max label + 1 to avoid conflicts
                next_new_label = mask.max() + 1
                
                print(f"Splitting label {target_label}...")
                kept_components = 0
                for comp_id in range(1, num_features + 1):
                    comp_mask = (cc == comp_id)
                    comp_size = np.sum(comp_mask)
                    
                    if comp_size < min_size:
                        # Remove small components
                        processed_mask[comp_mask] = 0
                    else:
                        # Assign a new, non-conflicting label to the valid component
                        processed_mask[comp_mask] = next_new_label
                        next_new_label += 1
                        kept_components += 1
                
                print(f"Label {target_label} was split into {kept_components} new labels.")

    # Save the processed mask file
    print(f"The maximum detected label value is: {np.max(processed_mask)}")
    tiff.imwrite(output_path, processed_mask.astype(mask.dtype))
    print(f"Processed mask has been saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a 3D TIFF mask file using connected component analysis and filter small components."
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input 3D TIFF mask file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the processed mask file.")
    parser.add_argument("-l", "--label", type=int, required=True, help="The target label to process. Use 0 to process all labels.")
    parser.add_argument("-m", "--min_size", type=int, required=True, help="Minimum size (in voxels) for connected components to be kept.")
    
    args = parser.parse_args()
    process_mask(args.input, args.output, args.label, args.min_size)