#!/usr/bin/env python3
"""
过滤预测结果脚本
将pred中不在mask区域的像素过滤掉（设为0）

用法:
    python filter_pred_by_mask.py --pred <pred_file> --mask <mask_file> --output <output_file>
    
参数:
    --pred: 预测结果文件路径 (支持.tiff, .tif, .npy, .h5格式)
    --mask: 掩码文件路径 (支持.tiff, .tif, .npy, .h5格式)
    --output: 输出文件路径 (可选，默认在原文件名后加_filtered)
    --threshold: mask阈值，大于此值的像素被认为是有效区域 (默认: 0)
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False
    print("Warning: tifffile not available. Install with: pip install tifffile")

try:
    import h5py
    H5_AVAILABLE = True
except ImportError:
    H5_AVAILABLE = False
    print("Warning: h5py not available. Install with: pip install h5py")


def load_image(file_path):
    """
    加载图像文件，支持多种格式
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        numpy.ndarray: 图像数组
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tiff', '.tif']:
        if not TIFF_AVAILABLE:
            raise ImportError("需要安装tifffile库来读取TIFF文件: pip install tifffile")
        return tifffile.imread(str(file_path))
    
    elif suffix == '.npy':
        return np.load(file_path)
    
    elif suffix == '.h5':
        if not H5_AVAILABLE:
            raise ImportError("需要安装h5py库来读取H5文件: pip install h5py")
        with h5py.File(file_path, 'r') as f:
            # 尝试读取常见的数据集名称
            for key in ['data', 'image', 'pred', 'mask']:
                if key in f:
                    return f[key][:]
            # 如果没找到常见名称，返回第一个数据集
            return f[list(f.keys())[0]][:]
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def save_image(data, file_path):
    """
    保存图像文件，支持多种格式
    
    Args:
        data (numpy.ndarray): 图像数据
        file_path (str): 输出文件路径
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # 确保输出目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if suffix in ['.tiff', '.tif']:
        if not TIFF_AVAILABLE:
            raise ImportError("需要安装tifffile库来保存TIFF文件: pip install tifffile")
        tifffile.imwrite(str(file_path), data, compression="zlib")
    
    elif suffix == '.npy':
        np.save(file_path, data)
    
    elif suffix == '.h5':
        if not H5_AVAILABLE:
            raise ImportError("需要安装h5py库来保存H5文件: pip install h5py")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
    
    else:
        raise ValueError(f"不支持的输出文件格式: {suffix}")


def filter_pred_by_mask(pred, mask, threshold=0):
    """
    根据mask过滤pred，将不在mask区域的像素设为0
    
    Args:
        pred (numpy.ndarray): 预测结果
        mask (numpy.ndarray): 掩码
        threshold (float): mask阈值，大于此值的像素被认为是有效区域
        
    Returns:
        numpy.ndarray: 过滤后的预测结果
    """
    # 确保pred和mask形状相同
    if pred.shape != mask.shape:
        raise ValueError(f"pred和mask形状不匹配: pred={pred.shape}, mask={mask.shape}")
    
    # 创建过滤后的pred副本
    filtered_pred = pred.copy()
    
    # 将mask中值小于等于threshold的像素在pred中设为0
    mask_valid = mask > threshold
    filtered_pred[~mask_valid] = 0
    
    return filtered_pred


def main():
    parser = argparse.ArgumentParser(description='根据mask过滤预测结果')
    parser.add_argument('--pred', required=True, help='预测结果文件路径')
    parser.add_argument('--mask', required=True, help='掩码文件路径')
    parser.add_argument('--output', help='输出文件路径 (可选)')
    parser.add_argument('--threshold', type=float, default=0, 
                       help='mask阈值，大于此值的像素被认为是有效区域 (默认: 0)')
    
    args = parser.parse_args()
    
    try:
        # 加载pred和mask
        print(f"加载预测结果: {args.pred}")
        pred = load_image(args.pred)
        print(f"预测结果形状: {pred.shape}, 数据类型: {pred.dtype}")
        
        print(f"加载掩码: {args.mask}")
        mask = load_image(args.mask)
        print(f"掩码形状: {mask.shape}, 数据类型: {mask.dtype}")
        
        # 过滤预测结果
        print(f"使用阈值 {args.threshold} 过滤预测结果...")
        filtered_pred = filter_pred_by_mask(pred, mask, args.threshold)
        
        # 计算统计信息
        original_nonzero = np.count_nonzero(pred)
        filtered_nonzero = np.count_nonzero(filtered_pred)
        mask_valid_pixels = np.count_nonzero(mask > args.threshold)
        
        print(f"原始预测非零像素数: {original_nonzero}")
        print(f"掩码有效像素数: {mask_valid_pixels}")
        print(f"过滤后非零像素数: {filtered_nonzero}")
        print(f"保留比例: {filtered_nonzero/original_nonzero*100:.2f}%")
        
        # 确定输出文件路径
        if args.output:
            output_path = args.output
        else:
            pred_path = Path(args.pred)
            output_path = pred_path.parent / f"{pred_path.stem}_filtered{pred_path.suffix}"
        
        # 保存过滤后的结果
        print(f"保存过滤后的结果到: {output_path}")
        save_image(filtered_pred, output_path)
        
        print("过滤完成!")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
