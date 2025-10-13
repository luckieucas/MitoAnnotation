import os
import argparse
import tifffile
import torch
import numpy as np
from tqdm import tqdm
import sys
import glob

from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing

# Import evaluation module
sys.path.append('/projects/weilab/liupeng/MitoAnnotation/src')
from evaluation.evaluate_res import evaluate_res

def main():
    parser = argparse.ArgumentParser(description="3D Mitochondria Segmentation Script")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input 3D TIFF image or directory.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the directory to save the output segmentations.")
    parser.add_argument("-c", "--config_path", type=str, default="MitoNet_v1.yaml", help="Path to the model config file.")
    parser.add_argument("--gt", type=str, default=None, help="Path to the ground truth TIFF image or directory for evaluation.")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode if ground truth is provided.")
    parser.add_argument("--downsampling", type=int, default=1, help="Image downsampling factor.")
    parser.add_argument("--confidence_thr", type=float, default=0.5, help="Segmentation confidence threshold.")
    parser.add_argument("--center_confidence_thr", type=float, default=0.1, help="Center confidence threshold.")
    parser.add_argument("--min_distance_object_centers", type=int, default=3, help="Minimum distance between object centers.")
    parser.add_argument("--min_size", type=int, default=500, help="Minimum size of objects in voxels.")
    parser.add_argument("--min_extent", type=int, default=5, help="Minimum bounding box extent for objects.")
    parser.add_argument("--pixel_vote_thr", type=int, default=2, help="Voxel vote threshold for ortho-plane consensus.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference if available.")
    args = parser.parse_args()

    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        # 查找所有tiff文件
        input_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
        if not input_files:
            raise ValueError(f"No TIFF files found in directory: {args.input}")
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    # 检查并创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Found {len(input_files)} file(s) to process")
    
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

    # 存储所有输出的consensus文件路径用于评估
    consensus_output_files = []
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # 加载图像
        print(f"Loading image from {input_file}...")
        image = tifffile.imread(input_file)
        
        # 检查图像维度
        if image.ndim != 3:
            print(f"Warning: Input image must be 3D, but got {image.ndim} dimensions. Skipping.")
            continue

        print(f"Image loaded with shape: {image.shape}")

        # 进行ortho-plane推断并保存每个平面的结果
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
                output_path = os.path.join(args.output, f"{base_filename}_{axis_name}.tif")
                print(f"  Saving {axis_name} segmentation to {output_path}...")
                tifffile.imwrite(output_path, plane_segmentation.astype(np.uint16), compression='zlib')
            else:
                print(f"  Could not generate segmentation for {axis_name} plane.")


        # 计算并保存最终共识
        print("\nGenerating final consensus segmentation...")
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
            output_path = os.path.join(args.output, f"{base_filename}_consensus.tif")
            print(f"Saving final consensus segmentation to {output_path}...")
            tifffile.imwrite(output_path, final_segmentation.astype(np.uint16), compression='zlib')
            consensus_output_files.append(output_path)
        else:
            print("Could not generate final consensus segmentation.")

    print("\n" + "="*60)
    print("All inference tasks completed!")
    print("="*60)
    
    # 如果启用评估且提供了ground truth
    if args.eval and args.gt:
        print("\n" + "="*60)
        print("Starting evaluation...")
        print("="*60)
        
        try:
            # 判断是单文件还是目录评估
            if os.path.isfile(args.gt) and len(consensus_output_files) == 1:
                # 单文件评估
                print(f"\nEvaluating single file:")
                print(f"  Prediction: {consensus_output_files[0]}")
                print(f"  Ground Truth: {args.gt}")
                
                metrics = evaluate_res(
                    pred_file=consensus_output_files[0],
                    gt_file=args.gt,
                    save_results=True
                )
                
                print("\n" + "-"*60)
                print("Evaluation Results:")
                print("-"*60)
                for key, value in metrics.items():
                    if key not in ["matched_pairs", "matched_scores", "pred_file", "gt_file", "file_name"]:
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                            
            elif os.path.isdir(args.gt) and os.path.isdir(args.output):
                # 目录评估
                print(f"\nEvaluating directory:")
                print(f"  Prediction Directory: {args.output}")
                print(f"  Ground Truth Directory: {args.gt}")
                
                from evaluation.evaluate_res import evaluate_directory
                results, summary = evaluate_directory(
                    pred_dir=args.output,
                    gt_dir=args.gt,
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
            else:
                print("\nWarning: Ground truth format does not match prediction format.")
                print("  - For single file: provide a single GT file")
                print("  - For directory: provide a GT directory with matching filenames")
                
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.eval and not args.gt:
        print("\nWarning: --eval flag is set but no ground truth (--gt) provided. Skipping evaluation.")
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)

if __name__ == "__main__":
    main()