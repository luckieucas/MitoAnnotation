import os
import argparse
import platform
import torch
import tifffile as tiff
import numpy as np
import sys
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
from empanada.config_loaders import load_config, read_yaml
from empanada_napari.utils import add_new_model
from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing

# 导入核心训练逻辑
from empanada_napari import finetune as finetune_logic

# 导入评估模块
sys.path.append('/projects/weilab/liupeng/MitoAnnotation/src')
from evaluation.evaluate_res import evaluate_res

def convert_3d_to_2d_slices(nnunet_dataset_path, output_path):
    """
    将nnUNet数据集的3D tiff文件转换为2D slices，按照empanada所需的格式组织
    
    Args:
        nnunet_dataset_path: nnUNet数据集路径，包含imagesTr, imagesTs, labelsTr, instancesTs等文件夹
        output_path: 输出路径，将创建train和val文件夹
    """
    print(f"Converting nnUNet dataset from {nnunet_dataset_path} to 2D slices...")
    
    # 处理训练集
    print("Processing training data...")
    imagesTr_path = os.path.join(nnunet_dataset_path, 'imagesTr')
    labelsTr_path = os.path.join(nnunet_dataset_path, 'instancesTr')

    train_images = sorted(glob(os.path.join(imagesTr_path, '*.tif*')))
    train_labels = sorted(glob(os.path.join(labelsTr_path, '*.tif*')))
    
    print(f"Found {len(train_images)} training images and {len(train_labels)} training labels")
    
    for img_path in train_images:
        # 读取3D图像
        img_3d = tiff.imread(img_path)
        base_name = os.path.basename(img_path).replace('_0000', '').replace('.tiff', '').replace('.tif', '')
        
        # 找到对应的标签文件
        label_path = None
        for lbl_path in train_labels:
            if base_name in os.path.basename(lbl_path):
                label_path = lbl_path
                break
        
        if label_path is None:
            print(f"Warning: No label found for {base_name}, skipping...")
            continue
        
        label_3d = tiff.imread(label_path)
        
        # 确保图像和标签的形状匹配
        if img_3d.shape != label_3d.shape:
            print(f"Warning: Shape mismatch for {base_name}: img {img_3d.shape} vs label {label_3d.shape}, skipping...")
            continue
        
        # 为每个volume创建子目录（empanada期望这种结构）
        volume_train_dir = os.path.join(output_path, 'train', base_name)
        volume_images_dir = os.path.join(volume_train_dir, 'images')
        volume_masks_dir = os.path.join(volume_train_dir, 'masks')
        os.makedirs(volume_images_dir, exist_ok=True)
        os.makedirs(volume_masks_dir, exist_ok=True)
        
        # 将每个z切片保存为单独的2D tiff
        num_slices = img_3d.shape[0]
        print(f"  Converting {base_name}: {num_slices} slices")
        
        saved_slices = 0
        skipped_slices = 0
        
        for z in range(num_slices):
            slice_name = f"slice_{z:04d}.tif"
            
            # 检查标签切片是否全是背景
            label_slice = label_3d[z]
            if np.sum(label_slice > 0) < 200:
                skipped_slices += 1
                continue  # 跳过全背景的切片
            
            # 保存图像切片
            img_slice = img_3d[z]
            # 确保图像是uint8类型
            if img_slice.dtype != np.uint8:
                if img_slice.max() <= 255:
                    img_slice = img_slice.astype(np.uint8)
                else:
                    # 归一化到0-255
                    img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(volume_images_dir, slice_name), img_slice, compression='zlib')
            
            # 保存标签切片
            # 确保标签是int32类型（empanada期望的类型
            label_slice = label_slice.astype(np.uint16)
            tiff.imwrite(os.path.join(volume_masks_dir, slice_name), label_slice, compression='zlib')
            
            saved_slices += 1
        
        print(f"    Saved: {saved_slices} slices, Skipped: {skipped_slices} empty slices")
    
    # 处理测试集（用作验证集）
    print("Processing test/validation data...")
    imagesTs_path = os.path.join(nnunet_dataset_path, 'imagesTs')
    instancesTs_path = os.path.join(nnunet_dataset_path, 'instancesTs')
    
    test_images = sorted(glob(os.path.join(imagesTs_path, '*.tif*')))
    test_labels = sorted(glob(os.path.join(instancesTs_path, '*.tif*')))
    
    print(f"Found {len(test_images)} test images and {len(test_labels)} test labels")
    
    for img_path in test_images:
        # 读取3D图像
        img_3d = tiff.imread(img_path)
        base_name = os.path.basename(img_path).replace('_0000', '').replace('.tiff', '').replace('.tif', '')
        
        # 找到对应的标签文件
        label_path = None
        for lbl_path in test_labels:
            if base_name in os.path.basename(lbl_path):
                label_path = lbl_path
                break
        
        if label_path is None:
            print(f"Warning: No label found for {base_name}, skipping...")
            continue
        
        label_3d = tiff.imread(label_path)
        
        # 确保图像和标签的形状匹配
        if img_3d.shape != label_3d.shape:
            print(f"Warning: Shape mismatch for {base_name}: img {img_3d.shape} vs label {label_3d.shape}, skipping...")
            continue
        
        # 为每个volume创建子目录
        volume_val_dir = os.path.join(output_path, 'val', base_name)
        volume_images_dir = os.path.join(volume_val_dir, 'images')
        volume_labels_dir = os.path.join(volume_val_dir, 'masks')
        os.makedirs(volume_images_dir, exist_ok=True)
        os.makedirs(volume_labels_dir, exist_ok=True)
        
        # 将每个z切片保存为单独的2D tiff
        num_slices = img_3d.shape[0]
        print(f"  Converting {base_name}: {num_slices} slices")
        
        saved_slices = 0
        skipped_slices = 0
        
        for z in range(num_slices):
            slice_name = f"slice_{z:04d}.tif"
            
            # 检查标签切片前景是否大于阈值
            label_slice = label_3d[z]
            if np.sum(label_slice > 0) < 1000:
                skipped_slices += 1
                continue  # 跳过背景过小的切片
            
            # 保存图像切片
            img_slice = img_3d[z]
            # 确保图像是uint8类型
            if img_slice.dtype != np.uint8:
                if img_slice.max() <= 255:
                    img_slice = img_slice.astype(np.uint8)
                else:
                    # 归一化到0-255
                    img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(volume_images_dir, slice_name), img_slice, compression='zlib')
            
            # 保存标签切片
            # 确保标签是int32类型（empanada期望的类型）
            if label_slice.dtype == np.uint16:
                label_slice = label_slice.astype(np.int32)
            tiff.imwrite(os.path.join(volume_labels_dir, slice_name), label_slice, compression='zlib')
            
            saved_slices += 1
        
        print(f"    Saved: {saved_slices} slices, Skipped: {skipped_slices} empty slices")
    
    print(f"Conversion complete! Data saved to {output_path}")
    # 使用递归glob统计文件数
    print(f"  Train images: {len(glob(os.path.join(output_path, 'train', '**/images/*.tif*'), recursive=True))}")
    print(f"  Train masks: {len(glob(os.path.join(output_path, 'train', '**/masks/*.tif*'), recursive=True))}")
    print(f"  Val images: {len(glob(os.path.join(output_path, 'val', '**/images/*.tif*'), recursive=True))}")
    print(f"  Val labels: {len(glob(os.path.join(output_path, 'val', '**/masks/*.tif*'), recursive=True))}")

def predict_3d(model_config_path, input_path, output_dir, args):
    """
    使用训练好的模型对3D TIFF文件进行预测
    
    Args:
        model_config_path: 训练好的模型配置文件路径 (.yaml)
        input_path: 输入3D TIFF文件或目录
        output_dir: 输出目录
        args: 命令行参数
    """
    print(f"\n{'='*60}")
    print("Starting 3D Prediction...")
    print(f"{'='*60}")
    
    # 检查输入是文件还是目录
    if os.path.isfile(input_path):
        input_files = [input_path]
    elif os.path.isdir(input_path):
        # 查找所有tiff文件
        input_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            input_files.extend(glob(os.path.join(input_path, ext)))
        if not input_files:
            raise ValueError(f"No TIFF files found in directory: {input_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(input_files)} file(s) to process")
    
    # 检查GPU是否可用
    use_gpu_flag = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not use_gpu_flag:
        print("GPU not available, falling back to CPU.")
    elif use_gpu_flag:
        print("Using GPU for inference.")
    else:
        print("Using CPU for inference.")
    
    # 加载模型配置
    print(f"Loading model configuration from {model_config_path}...")
    model_config = read_yaml(model_config_path)
    
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
    
    print(f"\nUsing axes for inference: {', '.join(args.axes)}")
    
    # 处理每个输入文件
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # 加载图像
        print(f"Loading image from {input_file}...")
        image = tiff.imread(input_file)
        
        # 检查图像维度
        if image.ndim != 3:
            print(f"Warning: Input image must be 3D, but got {image.ndim} dimensions. Skipping.")
            continue
        
        print(f"Image loaded with shape: {image.shape}")
        
        # 进行ortho-plane推断
        trackers_dict = {}
        
        # 获取需要调用的原始生成器函数
        stack_postprocessing_generator = stack_postprocessing.__wrapped__
        
        for axis_name in args.axes:
            print(f"Running inference on {axis_name} plane...")
            _, trackers = engine.infer_on_axis(image, axis_name)
            trackers_dict[axis_name] = trackers
            
            # 为当前平面后处理并生成分割体
            print(f"Post-processing and saving result for {axis_name} plane...")
            plane_postprocess_worker = stack_postprocessing_generator(
                {axis_name: trackers},
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
                # 去掉_0000后缀（如果存在）
                clean_filename = base_filename.replace('_0000', '')
                output_path = os.path.join(output_dir, f"{clean_filename}_{axis_name}.tiff")
                print(f"  Saving {axis_name} segmentation to {output_path}...")
                tiff.imwrite(output_path, plane_segmentation.astype(np.uint16), compression='zlib')
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
            # 去掉_0000后缀（如果存在）
            clean_filename = base_filename.replace('_0000', '')
            output_path = os.path.join(output_dir, f"{clean_filename}.tiff")
            print(f"Saving final consensus segmentation to {output_path}...")
            tiff.imwrite(output_path, final_segmentation.astype(np.uint16), compression='zlib')
        else:
            print("Could not generate final consensus segmentation.")
    
    print("\n" + "="*60)
    print("All prediction tasks completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

def evaluate_predictions(pred_dir, gt_dir, output_csv=None):
    """
    评估预测结果与ground truth
    
    Args:
        pred_dir: 预测结果目录（包含.tiff预测文件）
        gt_dir: ground truth目录（包含对应的实例分割文件）
        output_csv: 评估结果保存路径（CSV文件）
    
    Returns:
        DataFrame: 评估结果
    """
    print(f"\n{'='*60}")
    print("Starting Evaluation...")
    print(f"{'='*60}")
    print(f"Prediction directory: {pred_dir}")
    print(f"Ground truth directory: {gt_dir}")
    
    # 找到所有预测文件（.tiff格式）
    pred_files = sorted(glob(os.path.join(pred_dir, '*.tiff')))
    
    # 如果没有.tiff，尝试.tif
    if not pred_files:
        pred_files = sorted(glob(os.path.join(pred_dir, '*.tif')))
    
    if not pred_files:
        print("Warning: No prediction files found!")
        return None
    
    print(f"Found {len(pred_files)} prediction files")
    
    all_results = []
    
    for pred_file in pred_files:
        # 从预测文件名推断ground truth文件名
        base_name = os.path.basename(pred_file).replace('.tiff', '').replace('.tif', '')
        
        # 尝试不同的ground truth文件名格式
        gt_candidates = [
            os.path.join(gt_dir, f"{base_name}.tiff"),
            os.path.join(gt_dir, f"{base_name}.tif"),
            os.path.join(gt_dir, f"{base_name}_0000.tiff"),
        ]
        
        gt_file = None
        for candidate in gt_candidates:
            if os.path.exists(candidate):
                gt_file = candidate
                break
        
        if gt_file is None:
            print(f"Warning: No ground truth found for {base_name}, skipping...")
            continue
        
        print(f"\nEvaluating: {base_name}")
        print(f"  Prediction: {pred_file}")
        print(f"  Ground Truth: {gt_file}")
        
        try:
            # 使用evaluate_res进行评估
            metrics = evaluate_res(
                pred_file=pred_file,
                gt_file=gt_file,
                save_results=False  # 我们会统一保存结果
            )
            
            # 添加文件名信息
            metrics['file_name'] = base_name
            metrics['pred_file'] = pred_file
            metrics['gt_file'] = gt_file
            
            all_results.append(metrics)
            
            # 打印当前文件的结果
            print(f"  Results:")
            for key, value in metrics.items():
                if key not in ['matched_pairs', 'matched_scores', 'pred_file', 'gt_file', 'file_name']:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"  Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo successful evaluations!")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 计算平均指标
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['matched_pairs', 'matched_scores']
    metric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    mean_metrics = df[metric_cols].mean()
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"{'='*60}")
    print(f"Total files evaluated: {len(df)}")
    print("\nMean Metrics:")
    for key, value in mean_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"{'='*60}")
    
    # 保存结果
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to: {output_csv}")
        
        # 保存汇总结果
        summary_file = output_csv.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Evaluation Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Total files evaluated: {len(df)}\n\n")
            f.write("Mean Metrics:\n")
            for key, value in mean_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("="*60 + "\n")
        print(f"Summary saved to: {summary_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MitoNet with nnUNet dataset or predict with trained model.")
    
    # 添加模式选择
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'predict'], 
                       help="Mode: 'train' for training, 'predict' for prediction.")
    
    # 训练模式参数
    parser.add_argument("--dataset_path", type=str, nargs='?', default=None,
                       help="[Train mode] Path to the nnUNet dataset directory (e.g., /path/to/Dataset004_MitoHardCardiac).")
    parser.add_argument("--model_dir", type=str, nargs='?', default=None,
                       help="[Train mode] Directory to save the fine-tuned model; [Predict mode] Not used.")
    parser.add_argument("--output_data_path", type=str, default=None, 
                       help="[Train mode] Path to save converted 2D slices.")
    parser.add_argument("--model_config", type=str, default="MitoNet_v1.yaml", 
                       help="[Train mode] Path to the base model config file to fine-tune.")
    parser.add_argument("--model_name", type=str, default="FinetunedMitoNet", 
                       help="[Train mode] Name for the new fine-tuned model.")
    parser.add_argument("--iterations", type=int, default=1000, 
                       help="[Train mode] Number of training iterations.")
    parser.add_argument("--patch_size", type=int, default=256, 
                       help="[Train mode] Patch size in pixels for training.")
    parser.add_argument("--learning_rate", type=float, default=0.003, 
                       help="[Train mode] Maximum learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="[Train mode] Batch size for training.")
    parser.add_argument("--finetune_layer", type=str, default='all', 
                       choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'], 
                       help="[Train mode] Encoder layers to finetune.")
    parser.add_argument("--skip_conversion", action='store_true', 
                       help="[Train mode] Skip 3D to 2D conversion if data is already converted.")
    parser.add_argument("--test_after_training", action='store_true',
                       help="[Train mode] Automatically test on imagesTs and evaluate after training.")
    parser.add_argument("--test_axes", type=str, nargs='+', default=['xy', 'xz', 'yz'],
                       choices=['xy', 'xz', 'yz'],
                       help="[Train mode] Axes for testing. Default: xy xz yz")
    
    # 预测模式参数
    parser.add_argument("--trained_model", type=str, default=None,
                       help="[Predict mode] Path to trained model config file (.yaml).")
    parser.add_argument("--input", type=str, default=None,
                       help="[Predict mode] Path to input 3D TIFF file or directory.")
    parser.add_argument("--output", type=str, default=None,
                       help="[Predict mode] Path to output directory for predictions.")
    
    # 预测模式的推理参数
    parser.add_argument("--use_gpu", action="store_true", 
                       help="[Predict mode] Use GPU for inference if available.")
    parser.add_argument("--axes", type=str, nargs='+', default=['xy', 'xz', 'yz'], 
                       choices=['xy', 'xz', 'yz'], 
                       help="[Predict mode] Axes to perform inference on. Default: xy xz yz")
    parser.add_argument("--downsampling", type=int, default=1, 
                       help="[Predict mode] Image downsampling factor.")
    parser.add_argument("--confidence_thr", type=float, default=0.5, 
                       help="[Predict mode] Segmentation confidence threshold.")
    parser.add_argument("--center_confidence_thr", type=float, default=0.1, 
                       help="[Predict mode] Center confidence threshold.")
    parser.add_argument("--min_distance_object_centers", type=int, default=3, 
                       help="[Predict mode] Minimum distance between object centers.")
    parser.add_argument("--min_size", type=int, default=500, 
                       help="[Predict mode] Minimum size of objects in voxels.")
    parser.add_argument("--min_extent", type=int, default=5, 
                       help="[Predict mode] Minimum bounding box extent for objects.")
    parser.add_argument("--pixel_vote_thr", type=int, default=2, 
                       help="[Predict mode] Voxel vote threshold for ortho-plane consensus.")
    
    args = parser.parse_args()
    
    # 根据模式执行不同的操作
    if args.mode == "predict":
        # 预测模式
        if not args.trained_model:
            parser.error("--trained_model is required in predict mode")
        if not args.input:
            parser.error("--input is required in predict mode")
        if not args.output:
            parser.error("--output is required in predict mode")
        
        predict_3d(args.trained_model, args.input, args.output, args)
        return
    
    # 训练模式（原有逻辑）
    if not args.dataset_path:
        parser.error("dataset_path is required in train mode")
    if not args.model_dir:
        parser.error("model_dir is required in train mode")

    # 设置输出数据路径
    if args.output_data_path is None:
        args.output_data_path = os.path.join(args.dataset_path, f"2d_slices")
    
    # 转换3D tiff到2D slices
    if not args.skip_conversion:
        convert_3d_to_2d_slices(args.dataset_path, args.output_data_path)
    else:
        print(f"Skipping conversion, using existing data at {args.output_data_path}")
    
    # 设置训练和验证目录
    train_dir = os.path.join(args.output_data_path, 'train')
    eval_dir = os.path.join(args.output_data_path, 'val')
    
    print("\nLoading base finetune configuration...")
    # 从 empanada_napari 加载默认的微调配置
    import empanada_napari
    base_dir = os.path.dirname(empanada_napari.__file__)
    main_config_path = os.path.join(base_dir, 'training/finetune_config.yaml')
    config = load_config(main_config_path)

    print(f"Loading model-specific config from {args.model_config}...")
    model_config = load_config(args.model_config)

    # --- 合并和覆盖配置 ---
    # 1. 将模型定义（MODEL）和微调参数（FINETUNE）从模型特定配置中加载
    config['MODEL'] = {}
    for k, v in model_config.items():
        if k != 'FINETUNE':
            config['MODEL'][k] = model_config[k]
        else:
            config[k] = model_config[k]

    # 2. 设置用户自定义的参数
    config['model_name'] = args.model_name
    config['TRAIN']['train_dir'] = train_dir
    config['TRAIN']['model_dir'] = args.model_dir
    config['EVAL']['eval_dir'] = eval_dir
    config['TRAIN']['finetune_layer'] = args.finetune_layer
    config['TRAIN']['batch_size'] = args.batch_size
    config['TRAIN']['schedule_params']['max_lr'] = args.learning_rate

    # 3. 根据图像数量和迭代次数计算epochs
    n_imgs = len(glob(os.path.join(train_dir, '**/images/*'), recursive=True))
    if not n_imgs:
        raise Exception(f"No images found in {os.path.join(train_dir, '**/images/*')}")
    
    bsz = config['TRAIN']['batch_size']
    if n_imgs < bsz:
        print(f"Warning: Number of images ({n_imgs}) is less than batch size ({bsz}). Setting batch size to {n_imgs}.")
        config['TRAIN']['batch_size'] = n_imgs
        bsz = n_imgs

    epochs = int(args.iterations // (n_imgs // bsz)) if (n_imgs // bsz) > 0 else args.iterations
    print(f"n_imgs: {n_imgs}, bsz: {bsz}, epochs: {epochs}")
    print(f"Found {n_imgs} images for training. Training for {epochs} epochs to approximate {args.iterations} iterations.")

    # 4. 更新配置中的 epochs 和 patch size
    if 'epochs' in config['TRAIN']['schedule_params']:
        config['TRAIN']['schedule_params']['epochs'] = epochs
    else:
        config['TRAIN']['epochs'] = epochs

    for aug in config['TRAIN']['augmentations']:
        for k in aug.keys():
            if ('height' in k or 'width' in k) and aug.get(k) is None:
                 aug[k] = args.patch_size

    config['TRAIN']['save_freq'] = max(1, epochs // 5)
    config['EVAL']['epochs_per_eval'] = max(1, epochs // 5)

    # 5. 填充度量（metrics）所需的标签
    for metric in config['TRAIN']['metrics'] + config['EVAL']['metrics']:
        if metric['metric'] in ['IoU', 'PQ']:
            metric['labels'] = config['MODEL']['labels']
        elif metric['metric'] in ['F1']:
            metric['labels'] = config['MODEL']['thing_list']

    # 6. 处理平台特定问题
    if platform.system() == 'Darwin':
        config['TRAIN']['workers'] = 0

    # --- 开始微调 ---
    print("\nStarting fine-tuning process...")
    finetune_logic.main(config)
    print("Fine-tuning finished!")

    # --- 注册新模型 ---
    # 微调后，empanada 会保存一个新的 .pth 文件和 .yaml 文件
    # 我们可以将这个新生成的 yaml 文件注册到 empanada 中，方便之后调用
    output_model_yaml = os.path.join(args.model_dir, args.model_name + '.yaml')
    if os.path.exists(output_model_yaml):
        print(f"\nRegistering new model '{args.model_name}'...")
        add_new_model(args.model_name, output_model_yaml)
        print("Model registered successfully. You can now use it in the prediction script.")
    else:
        print("Could not find the output model yaml. Skipping registration.")
    
    print(f"\nTraining complete! Model saved to {args.model_dir}")
    print(f"Converted data available at: {args.output_data_path}")
    
    # --- 训练后自动测试和评估 ---
    if args.test_after_training:
        print("\n" + "="*60)
        print("Starting automatic testing after training...")
        print("="*60)
        
        # 检查是否存在测试数据
        imagesTs_path = os.path.join(args.dataset_path, 'imagesTs')
        instancesTs_path = os.path.join(args.dataset_path, 'instancesTs')
        
        if not os.path.exists(imagesTs_path):
            print(f"Warning: imagesTs directory not found at {imagesTs_path}")
            print("Skipping automatic testing.")
        elif not os.path.exists(instancesTs_path):
            print(f"Warning: instancesTs directory not found at {instancesTs_path}")
            print("Skipping automatic testing.")
        elif not os.path.exists(output_model_yaml):
            print(f"Warning: Model yaml not found at {output_model_yaml}")
            print("Skipping automatic testing.")
        else:
            # 设置测试输出目录（保存在数据集目录下，与imagesTs并列）
            test_output_dir = os.path.join(args.dataset_path, 'imagesTs_mitoNet_FT')
            os.makedirs(test_output_dir, exist_ok=True)
            
            print(f"\nTest images: {imagesTs_path}")
            print(f"Ground truth: {instancesTs_path}")
            print(f"Test output: {test_output_dir}")
            
            # 创建一个简单的args对象用于predict_3d
            class TestArgs:
                def __init__(self):
                    self.use_gpu = torch.cuda.is_available()
                    self.axes = args.test_axes
                    self.downsampling = 1
                    self.confidence_thr = 0.5
                    self.center_confidence_thr = 0.1
                    self.min_distance_object_centers = 3
                    self.min_size = 500
                    self.min_extent = 5
                    self.pixel_vote_thr = 2
            
            test_args = TestArgs()
            
            # 执行预测
            try:
                print("\n" + "-"*60)
                print("Running predictions on test set...")
                print("-"*60)
                predict_3d(output_model_yaml, imagesTs_path, test_output_dir, test_args)
                
                # 执行评估
                print("\n" + "-"*60)
                print("Evaluating predictions...")
                print("-"*60)
                # 在imagesTs_mitoNet_FT目录下创建evaluation子目录
                eval_dir = os.path.join(test_output_dir, 'evaluation')
                os.makedirs(eval_dir, exist_ok=True)
                eval_csv = os.path.join(eval_dir, 'evaluation_results.csv')
                eval_results = evaluate_predictions(test_output_dir, instancesTs_path, eval_csv)
                
                if eval_results is not None:
                    print("\n✅ Testing and evaluation completed successfully!")
                    print(f"Evaluation results saved to: {eval_csv}")
                else:
                    print("\n⚠️ Evaluation completed with some issues.")
                    
            except Exception as e:
                print(f"\n❌ Error during automatic testing: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without automatic testing...")
    
    print("\n" + "="*60)
    print("All tasks completed!")
    print("="*60)

if __name__ == "__main__":
    main()

