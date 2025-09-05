import os
import argparse
import tifffile
import torch
import numpy as np
from tqdm import tqdm
import glob

from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing
from evaluate_res import evaluate_res

def main():
    parser = argparse.ArgumentParser(description="3D Mitochondria Segmentation Script for nnUNet Raw Directory")
    parser.add_argument("input_path", type=str, help="Path to the nnUNet raw directory containing datasets.")
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="List of dataset names to process (e.g., Dataset001 Dataset002).")
    parser.add_argument("-c", "--config_path", type=str, default="configs/MitoNet_v1.yaml", help="Path to the model config file.")
    parser.add_argument("--downsampling", type=int, default=1, help="Image downsampling factor.")
    parser.add_argument("--confidence_thr", type=float, default=0.5, help="Segmentation confidence threshold.")
    parser.add_argument("--center_confidence_thr", type=float, default=0.1, help="Center confidence threshold.")
    parser.add_argument("--min_distance_object_centers", type=int, default=3, help="Minimum distance between object centers.")
    parser.add_argument("--min_size", type=int, default=500, help="Minimum size of objects in voxels.")
    parser.add_argument("--min_extent", type=int, default=5, help="Minimum bounding box extent for objects.")
    parser.add_argument("--pixel_vote_thr", type=int, default=2, help="Voxel vote threshold for ortho-plane consensus.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference if available.")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after prediction.")
    parser.add_argument("--force_repredict", action="store_true", help="Force re-prediction even if prediction files exist.")
    args = parser.parse_args()

    # 检查GPU是否可用
    use_gpu_flag = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not use_gpu_flag:
        print("GPU not available, falling back to CPU.")

    # 加载模型配置
    print(f"Loading model configuration from {args.config_path}...")
    model_config = read_yaml(args.config_path)

    # 初始化3D推断引擎
    engine = Engine3d(
        model_config,
        inference_scale=args.downsampling,
        confidence_thr=args.confidence_thr,
        nms_threshold=args.center_confidence_thr,
        nms_kernel=args.min_distance_object_centers,
        min_size=args.min_size,
        min_extent=args.min_extent,
        use_gpu=use_gpu_flag,
    )

    # 处理每个数据集
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # 构建数据集路径
        dataset_path = os.path.join(args.input_path, dataset_name)
        imagesTs_path = os.path.join(dataset_path, "imagesTs")
        instancesTs_path = os.path.join(dataset_path, "instancesTs")
        output_path = os.path.join(dataset_path, "imagesTs_mitoNet_pred")
        
        # 检查路径是否存在
        if not os.path.exists(imagesTs_path):
            print(f"Warning: imagesTs directory not found for {dataset_name}: {imagesTs_path}")
            continue
            
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 查找所有tiff文件
        tiff_files = glob.glob(os.path.join(imagesTs_path, "*.tif*"))
        if not tiff_files:
            print(f"No tiff files found in {imagesTs_path}")
            continue
            
        print(f"Found {len(tiff_files)} tiff files to process")
        
        # 统计变量
        skipped_count = 0
        processed_count = 0
        error_count = 0
        
        # 处理每个tiff文件
        for tiff_file in tqdm(tiff_files, desc=f"Processing {dataset_name}"):
            base_filename = os.path.splitext(os.path.basename(tiff_file))[0]
            print(f"\nProcessing: {base_filename}")
            
            # 检查是否已经存在预测文件
            consensus_output_path = os.path.join(output_path, f"{base_filename}_consensus.tif")
            prediction_exists = os.path.exists(consensus_output_path)
            
            if prediction_exists and not args.force_repredict:
                print(f"Prediction already exists for {base_filename}, skipping prediction...")
                skipped_count += 1
            else:
                if prediction_exists and args.force_repredict:
                    print(f"Force re-prediction enabled, overwriting existing prediction for {base_filename}...")
                try:
                    # 加载图像
                    image = tifffile.imread(tiff_file)
                    
                    # 检查图像维度
                    if image.ndim != 3:
                        print(f"Warning: {base_filename} is not 3D (shape: {image.shape}), skipping...")
                        continue

                    print(f"Image loaded with shape: {image.shape}")

                    # 进行ortho-plane推断
                    trackers_dict = {}
                    
                    # 获取需要调用的原始生成器函数
                    stack_postprocessing_generator = stack_postprocessing.__wrapped__
                    
                    for axis_name in ['xy', 'xz', 'yz']:
                        print(f"Running inference on {axis_name} plane...")
                        _, trackers = engine.infer_on_axis(image, axis_name)
                        trackers_dict[axis_name] = trackers
                        
                        # 为当前平面后处理并生成分割体
                        print(f"Post-processing and saving result for {axis_name} plane...")
                        plane_postprocess_worker = stack_postprocessing_generator(
                            {axis_name: trackers}, # 只传入当前平面的tracker
                            store_url=None,
                            model_config=model_config,
                            min_size=args.min_size,
                            min_extent=args.min_extent,
                        )

                        plane_segmentation = None
                        for seg_vol, class_name, _ in plane_postprocess_worker:
                            print(f"  Generated stack for class '{class_name}' from {axis_name} plane.")
                            if plane_segmentation is None:
                                plane_segmentation = seg_vol
                        
                        if plane_segmentation is not None:
                            plane_output_path = os.path.join(output_path, f"{base_filename}_{axis_name}.tif")
                            print(f"  Saving {axis_name} segmentation to {plane_output_path}...")
                            tifffile.imwrite(plane_output_path, plane_segmentation.astype(np.uint16), compression='zlib')
                        else:
                            print(f"  Could not generate segmentation for {axis_name} plane.")

                    # 计算并保存最终共识
                    print("Generating final consensus segmentation...")
                    consensus_generator = tracker_consensus.__wrapped__
                    consensus_worker = consensus_generator(
                        trackers_dict,
                        store_url=None,
                        model_config=model_config,
                        pixel_vote_thr=args.pixel_vote_thr,
                        min_size=args.min_size,
                        min_extent=args.min_extent,
                    )
                    
                    final_segmentation = None
                    for consensus_vol, class_name, _ in consensus_worker:
                        print(f"Generated consensus for class '{class_name}'.")
                        if final_segmentation is None:
                            final_segmentation = consensus_vol

                    if final_segmentation is not None:
                        print(f"Saving final consensus segmentation to {consensus_output_path}...")
                        tifffile.imwrite(consensus_output_path, final_segmentation.astype(np.uint16), compression='zlib')
                        processed_count += 1
                    else:
                        print("Could not generate final consensus segmentation.")
                        error_count += 1
                        continue
                        
                except Exception as e:
                    print(f"Error processing {base_filename}: {e}")
                    error_count += 1
                    continue
            
            # 如果启用评估且存在ground truth，进行评估（无论是否跳过预测）
            if args.evaluate and os.path.exists(instancesTs_path):
                gt_file = os.path.join(instancesTs_path, f"{base_filename}.tiff").replace("_0000", "")
                if os.path.exists(gt_file):
                    print(f"Running evaluation for {base_filename}...")
                    try:
                        metrics = evaluate_res(pred_file=consensus_output_path, gt_file=gt_file)
                        print(f"Evaluation completed for {base_filename}")
                        print(f"F1 Score: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
                    except Exception as e:
                        print(f"Evaluation failed for {base_filename}: {e}")
                else:
                    print(f"Ground truth file not found: {gt_file}")
            else:
                if not args.evaluate:
                    print(f"Evaluation skipped for {base_filename} (--evaluate not specified)")
                else:
                    print(f"Evaluation skipped for {base_filename} (instancesTs directory not found)")
        
        # 打印数据集处理统计
        print(f"\n{'-'*40}")
        print(f"Dataset {dataset_name} processing summary:")
        print(f"  Total files: {len(tiff_files)}")
        print(f"  Skipped (already exist): {skipped_count}")
        print(f"  Processed: {processed_count}")
        print(f"  Errors: {error_count}")
        print(f"{'-'*40}")

    print("\nAll datasets processed!")

if __name__ == "__main__":
    main()