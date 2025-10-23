#!/usr/bin/env python3
"""
快速测试脚本：验证nnUNet数据集到2D切片的转换功能
不进行实际训练，仅测试数据转换和验证数据质量
"""

import os
import sys
import argparse
import tifffile as tiff
import numpy as np
from glob import glob
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.mitoNet_finetune_from_nnunet import convert_3d_to_2d_slices

def verify_conversion(output_path):
    """验证转换结果的质量"""
    print("\n" + "="*60)
    print("Verifying conversion results...")
    print("="*60)
    
    # 检查目录结构
    required_dirs = [
        'train/images',
        'train/masks',
        'val/raw',
        'val/labels'
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(output_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_name}")
            return False
        else:
            num_files = len(glob(os.path.join(dir_path, '*.tif*')))
            print(f"✓ {dir_name}: {num_files} files")
    
    # 验证数据质量
    print("\nVerifying data quality...")
    
    # 检查训练集
    train_images = sorted(glob(os.path.join(output_path, 'train/images/*.tif*')))
    train_masks = sorted(glob(os.path.join(output_path, 'train/masks/*.tif*')))
    
    if len(train_images) != len(train_masks):
        print(f"❌ Mismatch: {len(train_images)} images vs {len(train_masks)} masks in training set")
        return False
    
    # 随机检查几个样本
    import random
    samples_to_check = min(5, len(train_images))
    sample_indices = random.sample(range(len(train_images)), samples_to_check)
    
    print(f"\nChecking {samples_to_check} random samples from training set...")
    for idx in sample_indices:
        img_path = train_images[idx]
        mask_path = train_masks[idx]
        
        img = tiff.imread(img_path)
        mask = tiff.imread(mask_path)
        
        if img.shape != mask.shape:
            print(f"❌ Shape mismatch in {os.path.basename(img_path)}: {img.shape} vs {mask.shape}")
            return False
        
        # 检查是否是2D
        if len(img.shape) != 2:
            print(f"❌ Not 2D image: {os.path.basename(img_path)} has shape {img.shape}")
            return False
        
        print(f"  ✓ {os.path.basename(img_path)}: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min()}, {img.max()}], unique_labels={len(np.unique(mask))}")
    
    # 检查验证集
    val_images = sorted(glob(os.path.join(output_path, 'val/raw/*.tif*')))
    val_labels = sorted(glob(os.path.join(output_path, 'val/labels/*.tif*')))
    
    if len(val_images) != len(val_labels):
        print(f"❌ Mismatch: {len(val_images)} images vs {len(val_labels)} labels in validation set")
        return False
    
    print(f"\nChecking {min(3, len(val_images))} random samples from validation set...")
    sample_indices = random.sample(range(len(val_images)), min(3, len(val_images)))
    
    for idx in sample_indices:
        img_path = val_images[idx]
        label_path = val_labels[idx]
        
        img = tiff.imread(img_path)
        label = tiff.imread(label_path)
        
        if img.shape != label.shape:
            print(f"❌ Shape mismatch in {os.path.basename(img_path)}: {img.shape} vs {label.shape}")
            return False
        
        if len(img.shape) != 2:
            print(f"❌ Not 2D image: {os.path.basename(img_path)} has shape {img.shape}")
            return False
        
        print(f"  ✓ {os.path.basename(img_path)}: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min()}, {img.max()}], unique_labels={len(np.unique(label))}")
    
    print("\n" + "="*60)
    print("✓ All checks passed!")
    print("="*60)
    return True

def main():
    parser = argparse.ArgumentParser(description="Test nnUNet to 2D slices conversion")
    parser.add_argument("dataset_path", type=str, help="Path to nnUNet dataset")
    parser.add_argument("--output_path", type=str, default=None, 
                       help="Output path for 2D slices (default: dataset_path/../dataset_name_2d_test)")
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output_path is None:
        dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
        args.output_path = os.path.join(os.path.dirname(args.dataset_path), 
                                       f"{dataset_name}_2d_test")
    
    print("="*60)
    print("Testing nnUNet to 2D Slices Conversion")
    print("="*60)
    print(f"Input: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print("="*60)
    
    # 检查输入数据集
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        return 1
    
    required_input_dirs = ['imagesTr', 'imagesTs']
    for dir_name in required_input_dirs:
        dir_path = os.path.join(args.dataset_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Required directory not found: {dir_name}")
            return 1
        num_files = len(glob(os.path.join(dir_path, '*.tif*')))
        print(f"✓ Found {num_files} files in {dir_name}")
    
    # 执行转换
    print("\nStarting conversion...")
    try:
        convert_3d_to_2d_slices(args.dataset_path, args.output_path)
    except Exception as e:
        print(f"\n❌ Conversion failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 验证结果
    if verify_conversion(args.output_path):
        print("\n✅ Test completed successfully!")
        print(f"Converted data is ready at: {args.output_path}")
        print("\nYou can now use this data for training with:")
        print(f"  python src/training/mitoNet_finetune_from_nnunet.py \\")
        print(f"      {args.dataset_path} \\")
        print(f"      ./trained_models/model_name \\")
        print(f"      --output_data_path {args.output_path} \\")
        print(f"      --skip_conversion")
        return 0
    else:
        print("\n❌ Verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())



