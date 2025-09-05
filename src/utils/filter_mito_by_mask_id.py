#!/usr/bin/env python3
"""
根据指定的mask id过滤mito数据的脚本
读取mitoEM-H_den_mask.h5和mitoEM-H_den_mito.h5文件，
过滤掉不在指定mask id列表中的mito
"""

import os
import h5py
import numpy as np
import argparse
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_h5_file(file_path, dataset_key=None):
    """
    读取h5文件并返回数据
    
    Args:
        file_path: h5文件路径
        dataset_key: 数据集键名，如果为None则自动选择第一个
    
    Returns:
        numpy数组
    """
    try:
        with h5py.File(file_path, 'r') as f:
            logger.info(f"文件 {file_path} 中的数据集: {list(f.keys())}")
            
            if dataset_key and dataset_key in f:
                data = f[dataset_key][()]
                logger.info(f"使用指定的数据集键 '{dataset_key}'")
            else:
                # 自动选择第一个数据集
                first_key = list(f.keys())[0]
                data = f[first_key][()]
                logger.info(f"自动选择数据集 '{first_key}'")
            
            logger.info(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
            return data
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        raise

def filter_mito_by_mask_id(mask_data, mito_data, valid_mask_ids, output_path=None):
    """
    根据有效的mask id过滤mito数据
    
    Args:
        mask_data: mask数据数组
        mito_data: mito数据数组
        valid_mask_ids: 有效的mask id列表
        output_path: 输出文件路径，如果为None则不保存
    
    Returns:
        过滤后的mito数据
    """
    logger.info(f"开始过滤mito数据...")
    logger.info(f"原始mask数据形状: {mask_data.shape}")
    logger.info(f"原始mito数据形状: {mito_data.shape}")
    logger.info(f"有效的mask id: {valid_mask_ids}")
    
    # 创建mask id的布尔掩码
    valid_mask_mask = np.isin(mask_data, valid_mask_ids)
    logger.info(f"有效mask像素数量: {np.sum(valid_mask_mask)}")
    
    # 过滤mito数据：只保留在有效mask区域内的mito
    filtered_mito = mito_data.copy()
    filtered_mito[~valid_mask_mask] = 0
    
    # 统计过滤结果
    original_mito_count = np.sum(mito_data > 0)
    filtered_mito_count = np.sum(filtered_mito > 0)
    logger.info(f"原始mito像素数量: {original_mito_count}")
    logger.info(f"过滤后mito像素数量: {filtered_mito_count}")
    logger.info(f"过滤掉的mito像素数量: {original_mito_count - filtered_mito_count}")
    
    # 如果指定了输出路径，保存过滤后的数据
    if output_path:
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('data', data=filtered_mito, compression='gzip')
            logger.info(f"过滤后的数据已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存文件时出错: {e}")
            raise
    
    return filtered_mito

def main():
    parser = argparse.ArgumentParser(description="根据指定的mask id过滤mito数据")
    parser.add_argument("--mask_file", type=str, required=True, 
                       help="mask h5文件路径 (mitoEM-H_den_mask.h5)")
    parser.add_argument("--mito_file", type=str, required=True, 
                       help="mito h5文件路径 (mitoEM-H_den_mito.h5)")
    parser.add_argument("--mask_ids", type=int, nargs='+', required=True,
                       help="有效的mask id列表，例如: 1 2 3 4")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径，如果不指定则不保存")
    parser.add_argument("--mask_dataset", type=str, default=None,
                       help="mask文件中的数据集键名")
    parser.add_argument("--mito_dataset", type=str, default=None,
                       help="mito文件中的数据集键名")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.mask_file):
        logger.error(f"Mask文件不存在: {args.mask_file}")
        return
    
    if not os.path.exists(args.mito_file):
        logger.error(f"Mito文件不存在: {args.mito_file}")
        return
    
    try:
        # 读取数据
        logger.info("正在读取mask文件...")
        mask_data = read_h5_file(args.mask_file, args.mask_dataset)
        
        logger.info("正在读取mito文件...")
        mito_data = read_h5_file(args.mito_file, args.mito_dataset)
        
        # 检查数据形状是否匹配
        if mask_data.shape != mito_data.shape:
            logger.error(f"数据形状不匹配: mask={mask_data.shape}, mito={mito_data.shape}")
            return
        
        # 过滤数据
        filtered_mito = filter_mito_by_mask_id(
            mask_data, mito_data, args.mask_ids, args.output
        )
        
        logger.info("过滤完成！")
        
        # 显示一些统计信息
        unique_mask_ids = np.unique(mask_data)
        logger.info(f"mask数据中的唯一id: {unique_mask_ids}")
        logger.info(f"指定的有效id: {args.mask_ids}")
        logger.info(f"无效的id: {[id for id in unique_mask_ids if id not in args.mask_ids]}")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()

