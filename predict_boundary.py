import os
import argparse
import numpy as np
import torch
from scipy.ndimage import label, binary_fill_holes
import tifffile as tiff
from skimage.morphology import binary_erosion
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import sum_labels

# It is assumed that you have added micro_sam to your Python path.
from micro_sam.util import get_sam_model, precompute_image_embeddings, set_precomputed, get_centers_and_bounding_boxes
from micro_sam.prompt_based_segmentation import segment_from_points
from micro_sam.prompt_generators import PointAndBoxPromptGenerator
from joblib import Parallel, delayed

# 全局字典，用于在并行工作进程中存储和复用模型实例，避免重复加载
predictor_storage = {}


def segment_slice_and_get_boundary(
    predictor,
    image_slice,
    labels_slice,
    n_points_per_label=1,
    erosion_width=1,
    label_list=None
):
    """
    (已优化) Segments a single 2D slice and extracts the boundaries of the labeled regions.
    This version uses batched point prompts for efficiency.

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
    
    if len(labels_to_process) == 0:
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

        if points_coords.numel() == 0:
            continue
            
        points_coords = points_coords.numpy().squeeze(axis=0)
        point_labels = point_labels.numpy().squeeze(axis=0)

        # ✅ 优化点: 一次性将所有提示点传入模型
        # 使用 return_all=False 直接获取最佳掩码，避免处理多个返回结果
        mask, _, _ = segment_from_points(
            predictor,
            points_coords[:, ::-1],  # SAM 需要 (y, x) 顺序的坐标
            point_labels,
            return_all=False  # 直接返回最佳掩码, shape (1, H, W)
        )
        mask = mask.squeeze() # 将 (1, H, W) 变为 (H, W)
        
        # Find the boundary by XORing the mask with its eroded version.
        eroded_mask = binary_erosion(mask, footprint=np.ones((erosion_width, erosion_width)))
        boundary = mask ^ eroded_mask
        slice_boundaries_combined[boundary] = label_id
            
    return slice_boundaries_combined


def process_slice_worker(args_tuple):
    """
    一个用于并行处理的 "worker" 函数。
    它接收一个元组作为参数，以兼容 joblib。
    """
    # 1. 解包参数
    i, image_slice, labels_slice, model_type, n_points, erosion_width, label_list = args_tuple
    
    # 2. 在每个工作进程中独立初始化模型
    # 这样可以避免在进程间传递庞大的模型对象
    pid = os.getpid()
    if pid not in predictor_storage:
        print(f"工作进程 {pid}: 正在初始化 SAM 模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor_storage[pid] = get_sam_model(model_type=model_type, device=device)
    
    predictor = predictor_storage[pid]

    # 3. 如果切片没有标签，则直接返回空结果
    if np.max(labels_slice) == 0:
        return i, np.zeros_like(labels_slice, dtype=np.uint8)

    # 4. 调用（已优化的）切片分割函数
    slice_boundaries = segment_slice_and_get_boundary(
        predictor,
        image_slice,
        labels_slice,
        n_points_per_label=n_points,
        erosion_width=erosion_width,
        label_list=label_list
    )
    
    return i, slice_boundaries


def process_volume_for_one_dimension(
    image_volume,
    label_volume,
    model_type, # 注意：现在传递 model_type 而不是 predictor
    axis,
    n_points_per_label,
    erosion_width,
    label_list=None
):
    """
    (已优化) Processes a 3D volume along a single dimension using parallel workers.
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

    # ✅ 优化点: 使用 joblib 并行处理所有切片
    axis_name = ['Z', 'Y', 'X'][axis]
    print(f"正在并行处理轴 {axis_name} 的 {n_slices} 个切片...")
    
    # 创建任务列表
    tasks = [
        (i, img_transposed[i], lbl_transposed[i], model_type, n_points_per_label, erosion_width, label_list)
        for i in range(n_slices)
    ]

    # n_jobs=-1 表示使用所有可用的 CPU 核心, backend="multiprocessing" 是健壮的选择
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(process_slice_worker)(task) for task in tqdm(tasks, desc=f"处理轴 {axis_name}")
    )

    # 将并行处理的结果重新组合成 3D 体数据
    for i, slice_boundaries in results:
        volume_boundaries_transposed[i] = slice_boundaries
    
    # Transpose the result back to the original orientation if needed.
    if axis == 0:
        return volume_boundaries_transposed
    else:
        original_order = np.argsort(axes_order)
        return np.transpose(volume_boundaries_transposed, original_order)

def find_holes_for_label(label_val, multi_label_mask, background_label):
    """
    Worker function: 对单个标签执行完整的孔洞填充逻辑。
    """
    # 1. 为当前标签创建二值掩码
    binary_mask_for_label = (multi_label_mask == label_val)
    
    # 2. 对这个二值掩码进行逐切片孔洞填充
    filled_binary_mask = fill_holes_3d_slice_by_slice(binary_mask_for_label)
    
    # 3. 找到新填充的、原先是背景的区域
    hole_locations = filled_binary_mask & (multi_label_mask == background_label)
    
    # 4. 返回孔洞位置和对应的标签值
    return hole_locations, label_val

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
    filled_mask_final = np.copy(multi_label_mask)
    unique_labels = np.unique(multi_label_mask)
    labels_to_process = np.delete(unique_labels, np.where(unique_labels == background_label))

    if not labels_to_process.any():
        print("Warning: No foreground labels found in the mask. Returning the original mask.")
        return filled_mask_final

    print(f"并行处理 {len(labels_to_process)} 个标签的孔洞填充...")

    # ✅ 并行计算阶段：
    # 并行地为所有标签找出它们的孔洞位置
    results = Parallel(n_jobs=-1)(
        delayed(find_holes_for_label)(lbl, multi_label_mask, background_label) 
        for lbl in labels_to_process
    )

    # ✅ 串行写入阶段：
    # 这个循环非常快，因为它只做赋值操作
    print("正在合并结果...")
    for hole_locations, label_val in results:
        if np.any(hole_locations): # 仅在找到孔洞时写入
            filled_mask_final[hole_locations] = label_val
            
    return filled_mask_final


def find_valid_components(lbl, mask_volume, min_size):
    """
    Worker function: Finds all valid connected components for a single original label.
    This version is optimized using vectorized SciPy/NumPy operations.
    """
    roi = (mask_volume == lbl)
    cc, num_features = label(roi)
    valid_components = []

    if num_features > 0:
        # 1. 一次性计算所有连通分量的大小
        # `sum_labels` 比在循环中调用 np.sum 快得多
        component_sizes = sum_labels(roi, labels=cc, index=np.arange(1, num_features + 1))
        
        # 2. 找出所有尺寸符合条件的标签ID
        # `np.where` 返回一个元组，我们取第一个元素
        valid_indices = np.where(component_sizes >= min_size)[0]
        
        # 3. 如果有符合条件的组件，再生成它们的掩码
        # 注意：我们只为通过筛选的组件创建掩码，避免不必要的工作
        if valid_indices.size > 0:
            # `valid_indices` 是从0开始的，但我们的标签是从1开始的，所以要+1
            valid_labels = valid_indices + 1
            for valid_lbl in valid_labels:
                valid_components.append(cc == valid_lbl)

    return valid_components

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
    
    # ✅ 步骤 1: 增加新的命令行参数
    parser.add_argument(
        '--merge_axes', 
        type=str, 
        nargs='*', 
        default=['z', 'y', 'x'], 
        help='Specify which axes results to merge. Use space-separated values. Can be "z", "y", "x" or "0", "1", "2". Default: z y x (all three).'
    )
    
    args = parser.parse_args()

    # ✅ 步骤 2: 解析 --merge_axes 参数，将 'z','y','x' 转换为 0,1,2
    axis_map = {'z': 0, 'y': 1, 'x': 2, '0': 0, '1': 1, '2': 2}
    try:
        # 使用 set 来自动处理重复的输入
        axes_to_process = sorted(list(set([axis_map[ax.lower()] for ax in args.merge_axes])))
    except KeyError as e:
        print(f"错误: 无效的轴标识符 {e}. 请使用 'z', 'y', 'x' 或 '0', '1', '2'.")
        exit()

    print(f"将要处理并合并以下轴的结果: {[ ['Z', 'Y', 'X'][i] for i in axes_to_process ]}")

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

    # --- Process Specified Dimensions ---
    # 使用字典来存储结果，而不是列表，这样更灵活
    boundary_results = {}
    
    # ✅ 步骤 3: 只循环处理用户指定的轴
    for axis in axes_to_process:
        axis_name = ['Z', 'Y', 'X'][axis]
        print(f"\n--- Starting processing for axis: {axis_name} ---")
        save_path = os.path.join(output_dir, f"generated_width_{args.erosion_width}_boundaries_from_{axis_name}_slices.tiff")
        
        if os.path.exists(save_path):
            print(f"Boundaries for axis {axis_name} already exist at {save_path}. Loading from file.")
            boundaries_for_axis = tiff.imread(save_path)
        else:
            boundaries_for_axis = process_volume_for_one_dimension(
                img_vol, mask_vol, args.model_type, axis, args.n_points, args.erosion_width, args.label_list
            )
            tiff.imwrite(save_path, boundaries_for_axis)
            print(f"Boundaries for axis {axis_name} saved to: {save_path}")
        
        # 将结果存入字典
        boundary_results[axis] = boundaries_for_axis

    # --- Merge, Filter, and Post-process ---
    print(f"\n--- Merging boundaries from specified axes: {[ ['Z', 'Y', 'X'][i] for i in axes_to_process ]} ---")

    # ✅ 步骤 4: 从字典中只取出需要合并的边界结果
    boundaries_to_merge = [boundary_results[axis] for axis in axes_to_process if axis in boundary_results]

    if not boundaries_to_merge:
        print("错误：没有找到任何可以合并的边界结果。程序将退出。")
        exit()
    
    # 使用筛选后的列表进行并集操作
    final_merged_boundaries = np.logical_or.reduce(boundaries_to_merge).astype(np.uint8)
    
    # --- 后续的所有处理逻辑保持不变 ---
    
    # Filter the original mask by removing the found boundaries.
    mask_vol[final_merged_boundaries > 0] = 0
    print("Post-processing: Splitting disconnected components and filtering by size.")
    
    # Create a new mask to store the re-labeled results.
    processed_mask = np.zeros_like(mask_vol, dtype=np.uint16)
    
    # Get all unique labels except for the background (0).
    unique_labels = np.unique(mask_vol)
    unique_labels = unique_labels[unique_labels != 0]
    
    print(f"Found {len(unique_labels)} unique labels remaining to process.")
    
    all_valid_components_nested = Parallel(n_jobs=-1)(
        delayed(find_valid_components)(lbl, mask_vol, args.min_size) 
        for lbl in tqdm(unique_labels, desc="Finding components")
    )

    print("Flattening results...")
    all_components_flat = [
        component for sublist in all_valid_components_nested for component in sublist
    ]

    print(f"Re-labeling {len(all_components_flat)} components sequentially...")
    current_new_label = 1
    for comp_mask in tqdm(all_components_flat, desc="Assigning new labels"):
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
    # On Windows, multiprocessing requires this guard. It's good practice everywhere.
    main()