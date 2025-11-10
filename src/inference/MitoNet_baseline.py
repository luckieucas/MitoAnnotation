import os
import argparse
import tifffile
import torch
import numpy as np
from tqdm import tqdm
import sys
import glob
import yaml


sys.path.append('/projects/weilab/liupeng/MitoAnnotation/src')
from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

# Import evaluation module

from evaluation.evaluate_res import evaluate_res

def load_config_yaml(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="3D Mitochondria Segmentation Script")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. If provided, loads parameters from config.")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                        help="Dataset name or ID (e.g., Dataset001_MitoHardBeta or 1).")
    parser.add_argument("-c", "--config_path", type=str, default=None,
                        help="Path to the model config file.")
    parser.add_argument("--eval", action="store_true", default=None,
                        help="Enable evaluation mode if ground truth is provided.")
    parser.add_argument("--downsampling", type=int, default=None,
                        help="Image downsampling factor.")
    parser.add_argument("--confidence_thr", type=float, default=None,
                        help="Segmentation confidence threshold.")
    parser.add_argument("--center_confidence_thr", type=float, default=None,
                        help="Center confidence threshold.")
    parser.add_argument("--min_distance_object_centers", type=int, default=None,
                        help="Minimum distance between object centers.")
    parser.add_argument("--min_size", type=int, default=None,
                        help="Minimum size of objects in voxels.")
    parser.add_argument("--min_extent", type=int, default=None,
                        help="Minimum bounding box extent for objects.")
    parser.add_argument("--pixel_vote_thr", type=int, default=None,
                        help="Voxel vote threshold for ortho-plane consensus.")
    parser.add_argument("--use_gpu", action="store_true", default=None,
                        help="Use GPU for inference if available.")
    parser.add_argument("--axes", type=str, nargs='+', default=None,
                        choices=['xy', 'xz', 'yz'],
                        help="Axes to perform inference on. Default: xy xz yz")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config_yaml(args.config)
        
        # Apply config values (command line args override config file)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    
    # Set default values if still None
    if args.dataset is None:
        parser.error("--dataset or --config (with dataset specified) is required")
    if args.config_path is None:
        args.config_path = "configs/MitoNet_v1.yaml"
    if args.eval is None:
        args.eval = False
    if args.downsampling is None:
        args.downsampling = 1
    if args.confidence_thr is None:
        args.confidence_thr = 0.5
    if args.center_confidence_thr is None:
        args.center_confidence_thr = 0.1
    if args.min_distance_object_centers is None:
        args.min_distance_object_centers = 3
    if args.min_size is None:
        args.min_size = 500
    if args.min_extent is None:
        args.min_extent = 5
    if args.pixel_vote_thr is None:
        args.pixel_vote_thr = 2
    if args.use_gpu is None:
        args.use_gpu = False
    if args.axes is None:
        args.axes = ['xy', 'xz', 'yz']

    # 转换 dataset name 或 ID 为标准 dataset name
    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    
    # 构建数据集根目录
    dataset_root = os.path.join(nnUNet_raw, dataset_name)
    if not os.path.exists(dataset_root):
        raise ValueError(f"Dataset directory does not exist: {dataset_root}")
    
    # 设置输入和输出路径
    input_dir = os.path.join(dataset_root, "imagesTs")
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_dir = os.path.join(dataset_root, "imagesTs_mitoNet_ZS")
    gt_dir = os.path.join(dataset_root, "instancesTs")
    mask_dir = os.path.join(dataset_root, "masksTs")
    
    # 查找所有tiff文件
    input_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))
    if not input_files:
        raise ValueError(f"No TIFF files found in directory: {input_dir}")
    
    # 检查并创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"nnUNet_raw: {nnUNet_raw}")
    print(f"Dataset input: {args.dataset}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
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
    
    print(f"\nUsing axes for inference: {', '.join(args.axes)}")
    
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
        
        for axis_name in args.axes:
            print(f"Running inference on {axis_name} plane...")
            _, trackers = engine.infer_on_axis(image, axis_name)
            trackers_dict[axis_name] = trackers
            
            # 为当前平面后处理并生成分割体
            print(f"Post-processing and saving result for {axis_name} plane...")
            plane_postprocess_worker = stack_postprocessing(
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
                output_path = os.path.join(output_dir, f"{base_filename}_{axis_name}.tiff")
                print(f"  Saving {axis_name} segmentation to {output_path}...")
                tifffile.imwrite(output_path, plane_segmentation.astype(np.uint16), compression='zlib')
            else:
                print(f"  Could not generate segmentation for {axis_name} plane.")


        # 计算并保存最终共识
        print("\nGenerating final consensus segmentation...")
        consensus_worker = tracker_consensus(
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
            output_path = os.path.join(output_dir, f"{base_filename}.tiff").replace("_0000", "")
            print(f"Saving final consensus segmentation to {output_path}...")
            tifffile.imwrite(output_path, final_segmentation.astype(np.uint16), compression='zlib')
            consensus_output_files.append(output_path)
        else:
            print("Could not generate final consensus segmentation.")

    print("\n" + "="*60)
    print("All inference tasks completed!")
    print("="*60)
    
    # 如果启用评估
    if args.eval:
        print("\n" + "="*60)
        print("Starting evaluation...")
        print("="*60)
        
        # 检查ground truth目录是否存在
        if not os.path.exists(gt_dir):
            print(f"\nWarning: Ground truth directory does not exist: {gt_dir}")
            print("Skipping evaluation.")
        else:
            try:
                # 目录评估
                print(f"\nEvaluating directory:")
                print(f"  Prediction Directory: {output_dir}")
                print(f"  Ground Truth Directory: {gt_dir}")
                
                from evaluation.evaluate_res import evaluate_directory
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

if __name__ == "__main__":
    main()