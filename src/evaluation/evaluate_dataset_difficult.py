# -*- coding: utf-8 -*-
"""
This script analyzes 3D mitochondrial masks from HDF5 files.
DCI calculation has been removed.
EFI fragment threshold is now 5% of the instance's volume.
MCI, MBI, surface area, and skeleton-related metrics have been removed.
Added Normalized Contact Count.
MODIFIED: Added timing for performance analysis.
FIXED: Handle disk space issues by using temp_folder and max_nbytes.
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import tempfile

from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.spatial.distance import cdist
from scipy.ndimage import label, generate_binary_structure, binary_dilation, binary_erosion

import matplotlib.pyplot as plt
import seaborn as sns


def process_single_mito(local_neighborhood_mask, center_label_id, compute_contact_count=False):
    """
    Performs all intra-organelle computations on a single mitochondrion.
    EFI threshold is now dynamic (5% of volume).
    REMOVED: DCI, MCI, MBI, surface area, and skeleton metrics are no longer calculated.
    """
    t_start_total = time.perf_counter()

    this_mask_in_neighborhood = (local_neighborhood_mask == center_label_id)
    props_in_patch = measure.regionprops(this_mask_in_neighborhood.astype(np.uint8))
    if not props_in_patch: return None
    
    cropped_mask = props_in_patch[0].image

    volume = np.sum(cropped_mask)
    
    structure = generate_binary_structure(rank=3, connectivity=3)
    
    # --- OPTIMIZATION: Perform erosion once and reuse the result ---
    t_start_erosion = time.perf_counter()
    eroded_mask = binary_erosion(cropped_mask, structure=structure)
    t_end_erosion = time.perf_counter()
    time_erosion_ms = (t_end_erosion - t_start_erosion) * 1000
    
    # --- EFI Calculation using the pre-computed eroded_mask ---
    t_start_efi = time.perf_counter()
    labeled_fragments, _ = label(eroded_mask, structure=structure)
    dynamic_efi_threshold = 0.05 * volume
    significant_fragments_mask = remove_small_objects(
        labeled_fragments, 
        min_size=int(dynamic_efi_threshold) + 1
    )
    efi = max(1,len(np.unique(significant_fragments_mask)) - 1)
    t_end_efi = time.perf_counter()
    time_efi_labeling_ms = (t_end_efi - t_start_efi) * 1000
    
    # --- Dilation-based Metrics with Normalized Contact Count ---
    contact_count = np.nan
    time_dilation_ms = 0
    if compute_contact_count:
        t_start_dilation = time.perf_counter()
        dilated = binary_dilation(this_mask_in_neighborhood, structure=structure)
        t_end_dilation = time.perf_counter()
        time_dilation_ms = (t_end_dilation - t_start_dilation) * 1000

        contact_region = local_neighborhood_mask[dilated]
        unique_labels = np.unique(contact_region)
        contact_labels = set(unique_labels) - {0, center_label_id}
        contact_count = len(contact_labels)

    t_end_total = time.perf_counter()
    time_total_ms = (t_end_total - t_start_total) * 1000

    results = {
        "mito_id": int(center_label_id),
        "volume_voxels3": volume,
        "efi": efi,
        "contact_count": contact_count,
        "nnd_voxels": np.nan,
        "time_total_ms": time_total_ms,
        "time_erosion_ms": time_erosion_ms,
        "time_efi_labeling_ms": time_efi_labeling_ms,
        "time_dilation_ms": time_dilation_ms,
    }
    
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            results[key] = round(value, 4)
            
    return results

def analyze_mito_mask_fast(mask_3d, compute_contact_count=False, temp_folder=None):
    t_start_analysis = time.perf_counter()

    if np.max(mask_3d) == 1: mask_3d, _ = label(mask_3d)
    props = measure.regionprops(mask_3d)
    tasks, PADDING, img_shape = [], 5, mask_3d.shape
    for prop in props:
        if compute_contact_count:
            min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
            pad_min_z=max(0,min_z-PADDING); pad_min_y=max(0,min_y-PADDING); pad_min_x=max(0,min_x-PADDING)
            pad_max_z=min(img_shape[0],max_z+PADDING); pad_max_y=min(img_shape[1],max_y+PADDING); pad_max_x=min(img_shape[2],max_x+PADDING)
            local_patch = mask_3d[pad_min_z:pad_max_z, pad_min_y:pad_max_y, pad_min_x:pad_max_x]
            tasks.append(delayed(process_single_mito)(local_patch, prop.label, compute_contact_count=True))
        else:
            fake_local_patch = prop.image.astype(np.uint8) * prop.label
            tasks.append(delayed(process_single_mito)(fake_local_patch, prop.label, compute_contact_count=False))
    
    # FIXED: Use threading backend to avoid disk space issues with process serialization
    # Threading is safer for large arrays and doesn't require disk serialization
    print(f"    Processing {len(tasks)} mitochondria using threading backend...")
    intra_results = Parallel(n_jobs=-1, backend='threading')(tasks)
    
    results_map = {r['mito_id']: r for r in intra_results if r is not None}
    if len(props) > 1:
        labels_arr = np.array([p.label for p in props])
        centroids_voxel = np.array([p.centroid for p in props])
        dist_matrix = cdist(centroids_voxel, centroids_voxel)
        np.fill_diagonal(dist_matrix, np.inf)
        nnd_values = dist_matrix.min(axis=1)
        for i, label_id in enumerate(labels_arr):
            if label_id in results_map: results_map[label_id]['nnd_voxels'] = round(nnd_values[i], 4)
    else:
        for label_id in results_map: results_map[label_id]['nnd_voxels'] = np.nan

    t_end_analysis = time.perf_counter()
    print(f"    File analysis complete in {t_end_analysis - t_start_analysis:.2f} seconds.")

    return list(results_map.values())


def process_folder(folder_path, output_dir, compute_contact_count=False):
    base_name = os.path.basename(os.path.normpath(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    detailed_csv_path = os.path.join(output_dir, f"{base_name}_details.csv")
    per_file_summary_path = os.path.join(output_dir, f"{base_name}_summary_per_file.csv")
    folder_summary_path = os.path.join(output_dir, f"{base_name}_summary_folder.csv")
    all_results_list, per_file_summary_list = [], []
    file_list = [f for f in os.listdir(folder_path) if f.endswith("_mito.h5") and "_crop" not in f]
    if not file_list: return None
    
    # Create a temp folder in a location with more space (if needed)
    temp_folder = tempfile.mkdtemp(dir=output_dir, prefix='joblib_tmp_')
    print(f"Using temporary folder: {temp_folder}")
    
    try:
        for filename in tqdm(file_list, desc=f"Processing {base_name}", leave=False):
            print(f"Processing file: {filename}")
            t_start_file = time.perf_counter()

            filepath = os.path.join(folder_path, filename)
            with h5py.File(filepath, 'r') as f: mask = f[list(f.keys())[0]][()]
            print(f"Mask shape: {mask.shape}")
            mito_results = analyze_mito_mask_fast(mask, compute_contact_count=compute_contact_count, 
                                                 temp_folder=temp_folder)
            if mito_results:
                df_file = pd.DataFrame(mito_results)
                file_avg = df_file.drop(columns=['mito_id']).mean(numeric_only=True).to_dict()
                file_avg['mito_count'] = len(df_file)
                file_avg['source_file'] = filename
                per_file_summary_list.append(file_avg)
                for r in mito_results:
                    r["file"] = filename
                    all_results_list.append(r)
            
            t_end_file = time.perf_counter()
            print(f"  > Finished processing file '{filename}' in {t_end_file - t_start_file:.2f} seconds.")
    finally:
        # Clean up temp folder
        import shutil
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Cleaned up temporary folder: {temp_folder}")

    if not all_results_list: return None
    df_detailed = pd.DataFrame(all_results_list)
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"-> Saved detailed results to: {detailed_csv_path}")
    if per_file_summary_list:
        df_per_file = pd.DataFrame(per_file_summary_list)
        df_per_file = df_per_file.round(4)
        time_cols = [c for c in df_per_file.columns if c.startswith('time_')]
        main_cols = [c for c in df_per_file.columns if c not in ['source_file', 'mito_count'] + time_cols]
        cols = ['source_file', 'mito_count']  + main_cols
        df_per_file = df_per_file[cols]
        df_per_file.to_csv(per_file_summary_path, index=False)
        print(f"-> Saved per-file summary to: {per_file_summary_path}")
    
    folder_avg_df = pd.DataFrame(per_file_summary_list)
    time_cols_to_drop = [c for c in folder_avg_df.columns if c.startswith('time_')]
    folder_avg_series = folder_avg_df.drop(columns=['source_file'] + time_cols_to_drop).mean(numeric_only=True)
    
    df_folder_avg = pd.DataFrame(folder_avg_series).transpose()
    df_folder_avg = df_folder_avg.round(4)
    cols = ['mito_count'] + [c for c in df_folder_avg.columns if c not in ['mito_count']]
    df_folder_avg = df_folder_avg[cols]
    df_folder_avg.to_csv(folder_summary_path, index=False)
    return folder_avg_series


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Mitochondrial metrics from HDF5 masks.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("root_folder", type=str, help="Path to the root folder containing experiment subdirectories")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save all output CSV files")
    parser.add_argument("--datasets-hard", type=str, nargs='+', default=None, help="A list of subfolder names for the 'hard' group (stars).")
    parser.add_argument("--datasets-easy", type=str, nargs='+', default=None, help="A list of subfolder names for the 'easy' group (circles).")
    parser.add_argument("--compute-contact-count", action='store_true', 
                        help="Enable calculation of the dilation-based Contact Count metric.")
    return parser.parse_args()


def main():
    args = parse_args()
    root_folder = args.root_folder
    output_dir = args.output_dir
    if not os.path.isdir(root_folder): print(f"Error: Provided path is not a directory: {root_folder}"); return
    print(f"Starting analysis for subfolders in: {root_folder}")
    datasets_hard = args.datasets_hard if args.datasets_hard else []
    datasets_easy = args.datasets_easy if args.datasets_easy else []
    if not datasets_hard and not datasets_easy:
        print("Warning: No datasets specified. Processing all found subfolders as 'easy' group.")
        all_subfolders_temp = {entry.name: entry.path for entry in os.scandir(root_folder) if entry.is_dir()}
        datasets_easy = list(all_subfolders_temp.keys())
    print(f"Hard datasets (star marker): {datasets_hard}")
    print(f"Easy datasets (circle marker): {datasets_easy}")
    if args.compute_contact_count:
        print("Dilation-based Contact Count metric is ENABLED.")
    print("EFI fragment threshold is dynamic: 5% of instance volume.")
    print(f"Results will be saved to: {output_dir}"); print("-" * 30)
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    print(f"Disk space in output directory: {free // (2**30)} GB free out of {total // (2**30)} GB total")
    if free < 10 * (2**30):  # Less than 10GB
        print("WARNING: Low disk space detected! Consider freeing up space or using a different output directory.")
    
    try: all_subfolders = {entry.name: entry.path for entry in os.scandir(root_folder) if entry.is_dir()}
    except FileNotFoundError: print(f"Error: Root folder not found at {root_folder}"); return
    folders_to_process, specified_datasets = [], datasets_hard + datasets_easy
    for name in specified_datasets:
        if name in all_subfolders: folders_to_process.append(all_subfolders[name])
        else: print(f"Warning: Specified dataset '{name}' not found. Skipping.")
    if not folders_to_process: print("No valid subdirectories found to process."); return
    all_folder_summaries = []
    print(f"Found {len(folders_to_process)} subdirectories to process.")
    for i, subfolder_path in enumerate(folders_to_process):
        print(f"\n--- Processing dataset {i+1}/{len(folders_to_process)}: {os.path.basename(subfolder_path)} ---")
        try:
            folder_summary = process_folder(subfolder_path, output_dir=output_dir, 
                                            compute_contact_count=args.compute_contact_count)
            if folder_summary is not None:
                folder_summary['subfolder_name'] = os.path.basename(subfolder_path)
                all_folder_summaries.append(folder_summary)
        except Exception as e:
            print(f"!!! An error occurred while processing {subfolder_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    print("\n" + "=" * 30); print("All processing complete.")
    if not all_folder_summaries: print("No data was generated, skipping plots."); return
    summary_df = pd.DataFrame(all_folder_summaries)
    summary_df.rename(columns={'subfolder_name': 'dataset'}, inplace=True)
    def get_group(dataset_name):
        if dataset_name in datasets_hard: return 'hard'
        elif dataset_name in datasets_easy: return 'easy'
        return 'easy'
    summary_df['group'] = summary_df['dataset'].apply(get_group)
    print("\nGenerating summary scatter plot...")
    if args.compute_contact_count:
        try:
            scatter_data = summary_df[['dataset', 'group', 'contact_count', 'efi']].copy()
            scatter_csv_filename = f"{os.path.basename(os.path.normpath(root_folder))}_efi_vs_contact_count_data.csv"
            scatter_csv_filepath = os.path.join(output_dir, scatter_csv_filename)
            scatter_data.to_csv(scatter_csv_filepath, index=False)
            print(f"-> Saved scatter plot data to CSV: {scatter_csv_filepath}")
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 12))
            hard_df = summary_df[summary_df['group'] == 'hard']
            easy_df = summary_df[summary_df['group'] == 'easy']
            
            if not hard_df.empty:
                hard_palette = sns.color_palette("YlOrRd_d", n_colors=len(hard_df['dataset'].unique()))
                sns.scatterplot(data=hard_df, x='contact_count', y='efi', hue='dataset', 
                                palette=hard_palette, marker='*', s=500, ax=ax, legend='full')
            if not easy_df.empty:
                easy_palette = sns.color_palette("GnBu_d", n_colors=len(easy_df['dataset'].unique()))
                sns.scatterplot(data=easy_df, x='contact_count', y='efi', hue='dataset', 
                                palette=easy_palette, marker='o', s=400, ax=ax, legend='full')
            
            max_val = max(summary_df['contact_count'].max(), summary_df['efi'].max()) if summary_df['contact_count'].notna().any() else summary_df['efi'].max()
            ax.set_xlim(0, max_val * 1.1); ax.set_ylim(0, max_val * 1.1)
            ax.set_aspect('equal', adjustable='box')
            
            ax.set_title('Comparison of Average EFI vs. Contact Count per Dataset', fontsize=16)
            ax.set_xlabel('Average Contact Count', fontsize=12)
            ax.set_ylabel('Average Erosion Fragility Index (EFI)', fontsize=12)
            ax.legend(title='Datasets'); plt.tight_layout()
            
            plot_filename = f"{os.path.basename(os.path.normpath(root_folder))}_efi_vs_contact_count_plot.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=300); plt.close()
            print(f"-> Successfully saved scatter plot to: {plot_filepath}")
        except Exception as e:
            print(f"!!! Could not generate scatter plot. Error: {e}")
    else:
        print("Skipping EFI vs Contact Count scatter plot because --compute-contact-count was not enabled.")
    print("\nGenerating bar charts for each metric...")
    bar_plot_dir = os.path.join(output_dir, "metric_comparison_charts")
    os.makedirs(bar_plot_dir, exist_ok=True)
    metric_columns = summary_df.select_dtypes(include=np.number).columns.tolist()
    metric_columns = [m for m in metric_columns if not m.startswith('time_')]

    for metric in metric_columns:
        if metric not in summary_df.columns or summary_df[metric].isnull().all(): continue
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=summary_df, x='dataset', y=metric, hue='group', dodge=False, palette={"hard": "#d65f5f", "easy": "#5f9ed6"})
            title_metric = metric.replace("_", " ").replace("voxels", "(voxels)").title()
            title = f'Comparison of {"Total" if metric == "mito_count" else "Average"} {title_metric} Across Datasets'
            ylabel = f'{"Total" if metric == "mito_count" else "Average"} {title_metric}'
            plt.title(title, fontsize=16); plt.xlabel('Dataset', fontsize=12); plt.ylabel(ylabel, fontsize=12)
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            plot_path = os.path.join(bar_plot_dir, f"summary_{metric}.png")
            plt.savefig(plot_path, dpi=300); plt.close()
        except Exception as e:
            print(f"!! Could not generate bar chart for '{metric}'. Error: {e}"); plt.close()
    print(f"-> Successfully saved all bar charts to: {bar_plot_dir}")


if __name__ == "__main__":
    main()