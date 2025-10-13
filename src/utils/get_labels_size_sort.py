import tifffile
import numpy as np
import argparse

def filter_and_sort_labels(mask_path, output_path,
                           size_threshold=200,
                           exclude_zero=True):
    """
    Reads a TIFF mask, removes labels with a voxel count below a threshold,
    and sorts the remaining labels by size.

    Args:
        mask_path (str):        File path of the input mask.
        output_path (str):      File path to save the filtered mask.
        size_threshold (int):   Threshold below which labels will be removed (set to 0).
        exclude_zero (bool):    Whether to exclude the label 0 (background) during counting and sorting.

    Returns:
        sorted_labels (List[int]):  A list of labels sorted in descending order by their voxel count after filtering.
        removed_labels (List[int]): A list of labels that were removed (set to 0).
        label_counts (Dict[int, int]): A dictionary mapping the remaining labels to their voxel counts.
    """
    # 1) Read the mask (supports 2D or 3D)
    print(f"Reading mask from: {mask_path}")
    mask = tifffile.imread(mask_path)

    # 2) Count the number of voxels for each label
    labels, counts = np.unique(mask, return_counts=True)
    label_counts = dict(zip(labels, counts))

    # 3) Optional: Exclude the background label 0
    if exclude_zero and 0 in label_counts:
        original_background_count = label_counts.pop(0)
        print(f"Excluding background label 0 (found {original_background_count} voxels).")
    
    initial_label_count = len(label_counts)
    print(f"Found {initial_label_count} non-background labels before filtering.")

    # 4) Identify labels to be removed
    removed_labels = [lbl for lbl, cnt in label_counts.items()
                        if cnt < size_threshold]

    # 5) Generate the filtered mask
    mask_filtered = mask.copy()
    if removed_labels:
        # Set all small labels to 0
        # Using isin is more efficient here
        mask_filtered[np.isin(mask_filtered, removed_labels)] = 0

    # 6) Re-count the size of the remaining labels after filtering, and sort them
    labels_remain, counts_remain = np.unique(mask_filtered, return_counts=True)
    counts_remain_map = dict(zip(labels_remain, counts_remain))
    if exclude_zero and 0 in counts_remain_map:
        counts_remain_map.pop(0)

    # Sort in descending order of voxel count
    sorted_labels = sorted(counts_remain_map.keys(),
                           key=lambda l: counts_remain_map[l],
                           reverse=True)
    
    print(f"Total labels before filtering: {initial_label_count}")
    print(f"Total labels after filtering: {len(sorted_labels)}")

    # 7) Save the filtered mask
    print(f"Saving filtered mask to: {output_path}")
    tifffile.imwrite(output_path, mask_filtered, compression='zlib')

    return sorted_labels, removed_labels, counts_remain_map


if __name__ == "__main__":
    # --- Argument Parsing Setup ---
    parser = argparse.ArgumentParser(
        description="Filter labels in a TIFF mask by size. This script removes any labeled region with a voxel count below a specified threshold and saves the result as a new TIFF file.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i","--input_path",
        type=str,
        help="Path to the input TIFF mask file."
    )


    parser.add_argument(
        "-s", "--size_threshold",
        type=int,
        default=200,
        help="Voxel count threshold. Labels smaller than this will be removed. (Default: 200)"
    )

    parser.add_argument(
        "--include_zero",
        action="store_true", # This flag's presence means the value is True
        help="Include the background label (0) in filtering and sorting statistics.\nBy default, label 0 is ignored."
    )
    
    args = parser.parse_args()
    
    # --- Function Execution ---
    # The 'exclude_zero' parameter is True by default in the function.
    # We set it to False only if the '--include_zero' flag is present.
    should_exclude_zero = not args.include_zero

    sorted_labels, removed_labels, counts = filter_and_sort_labels(
        mask_path=args.input_path,
        output_path=args.input_path.replace(".tiff", "_filtered_by_size.tiff"),
        size_threshold=args.size_threshold,
        exclude_zero=should_exclude_zero
    )

    # --- Print Summary ---
    print("\n--- Summary ---")
    print(f"Removed {len(removed_labels)} labels with voxel count < {args.size_threshold}.")
    if len(removed_labels) > 0:
      # Only print the list if it's not too long
      if len(removed_labels) < 20:
          print("Removed labels:", removed_labels)
      else:
          print(f"Removed labels: [{removed_labels[0]}, {removed_labels[1]}, ..., {removed_labels[-1]}]")


    print(f"\nFound {len(sorted_labels)} remaining labels sorted by size (largest -> smallest):")
    # Print top 10 largest labels as an example
    for i, lbl in enumerate(sorted_labels[:20]):
        print(f"  {i+1}. Label {lbl}: {counts[lbl]} voxels")
    if len(sorted_labels) > 20:
        print("  ...")