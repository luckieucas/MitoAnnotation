#!/usr/bin/env python3
"""
根据指定的mask id过滤mito数据的脚本
读取mask和mito的tiff文件，
过滤掉不在指定mask id列表中的mito
"""

import os
import tifffile
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

def read_tiff_file(file_path):
    """
    读取tiff文件并返回数据
    
    Args:
        file_path: tiff文件路径
    
    Returns:
        numpy数组
    """
    try:
        data = tifffile.imread(file_path)
        logger.info(f"成功读取文件: {file_path}")
        logger.info(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
        return data
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        raise

def filter_mito_by_mask_id(mask_data, mito_data, valid_mask_ids, 
                          output_mito_included=None, output_mito_excluded=None,
                          output_mask_included=None, output_mask_excluded=None):
    """
    根据有效的mask id过滤mito数据
    
    Args:
        mask_data: mask数据数组
        mito_data: mito数据数组
        valid_mask_ids: 有效的mask id列表
        output_mito_included: 输出包含指定id区域的mito文件路径，如果为None则不保存
        output_mito_excluded: 输出不包含指定id区域的mito文件路径，如果为None则不保存
        output_mask_included: 输出包含指定id的mask文件路径，如果为None则不保存
        output_mask_excluded: 输出不包含指定id的mask文件路径，如果为None则不保存
    
    Returns:
        包含指定id的mito数据, 不包含指定id的mito数据
    """
    logger.info(f"开始过滤mito数据...")
    logger.info(f"原始mask数据形状: {mask_data.shape}")
    logger.info(f"原始mito数据形状: {mito_data.shape}")
    logger.info(f"有效的mask id: {valid_mask_ids}")
    
    # 创建mask id的布尔掩码
    valid_mask_mask = np.isin(mask_data, valid_mask_ids)
    logger.info(f"有效mask像素数量: {np.sum(valid_mask_mask)}")
    
    # 生成两个mask
    # 1. 包含指定id的mask
    mask_included = mask_data.copy()
    mask_included[~valid_mask_mask] = 0
    
    # 2. 不包含指定id的mask（其余部分）
    mask_excluded = mask_data.copy()
    mask_excluded[valid_mask_mask] = 0
    
    # 生成两个mito数据
    # 1. 包含指定id区域的mito
    mito_included = mito_data.copy()
    mito_included[~valid_mask_mask] = 0
    
    # 2. 不包含指定id区域的mito（其余部分）
    mito_excluded = mito_data.copy()
    mito_excluded[valid_mask_mask] = 0
    
    # 统计过滤结果
    original_mito_count = np.sum(mito_data > 0)
    mito_included_count = np.sum(mito_included > 0)
    mito_excluded_count = np.sum(mito_excluded > 0)
    logger.info(f"原始mito像素数量: {original_mito_count}")
    logger.info(f"包含区域的mito像素数量: {mito_included_count}")
    logger.info(f"排除区域的mito像素数量: {mito_excluded_count}")
    
    # 统计mask信息
    unique_ids_included = np.unique(mask_included[mask_included > 0])
    unique_ids_excluded = np.unique(mask_excluded[mask_excluded > 0])
    logger.info(f"包含的mask中的唯一id数量: {len(unique_ids_included)}")
    logger.info(f"排除的mask中的唯一id数量: {len(unique_ids_excluded)}")
    
    # 保存包含指定id区域的mito
    if output_mito_included:
        try:
            tifffile.imwrite(output_mito_included, mito_included, compression='zlib')
            logger.info(f"包含区域的mito数据已保存到: {output_mito_included}")
        except Exception as e:
            logger.error(f"保存included mito文件时出错: {e}")
            raise
    
    # 保存不包含指定id区域的mito
    if output_mito_excluded:
        try:
            tifffile.imwrite(output_mito_excluded, mito_excluded, compression='zlib')
            logger.info(f"排除区域的mito数据已保存到: {output_mito_excluded}")
        except Exception as e:
            logger.error(f"保存excluded mito文件时出错: {e}")
            raise
    
    # 保存包含指定id的mask
    if output_mask_included:
        try:
            tifffile.imwrite(output_mask_included, mask_included, compression='zlib')
            logger.info(f"包含指定id的mask已保存到: {output_mask_included}")
        except Exception as e:
            logger.error(f"保存included mask文件时出错: {e}")
            raise
    
    # 保存不包含指定id的mask
    if output_mask_excluded:
        try:
            tifffile.imwrite(output_mask_excluded, mask_excluded, compression='zlib')
            logger.info(f"不包含指定id的mask已保存到: {output_mask_excluded}")
        except Exception as e:
            logger.error(f"保存excluded mask文件时出错: {e}")
            raise
    
    return mito_included, mito_excluded

def main():
    parser = argparse.ArgumentParser(description="根据指定的mask id过滤mito数据并生成included和excluded的mito和mask")
    parser.add_argument("--mask_file", type=str, required=True, 
                       help="mask tiff文件路径")
    parser.add_argument("--mito_file", type=str, required=True, 
                       help="mito tiff文件路径")
    parser.add_argument("--mask_ids", type=int, nargs='+', required=True,
                       help="有效的mask id列表，例如: 1 2 3 4")
    parser.add_argument("--output_mito_included", type=str, default=None,
                       help="输出包含指定id区域的mito文件路径，如果不指定则不保存")
    parser.add_argument("--output_mito_excluded", type=str, default=None,
                       help="输出不包含指定id区域的mito文件路径，如果不指定则不保存")
    parser.add_argument("--output_mask_included", type=str, default=None,
                       help="输出包含指定id的mask文件路径，如果不指定则不保存")
    parser.add_argument("--output_mask_excluded", type=str, default=None,
                       help="输出不包含指定id的mask文件路径，如果不指定则不保存")
    
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
        mask_data = read_tiff_file(args.mask_file)
        
        logger.info("正在读取mito文件...")
        mito_data = read_tiff_file(args.mito_file)
        
        # 检查数据形状是否匹配
        if mask_data.shape != mito_data.shape:
            logger.error(f"数据形状不匹配: mask={mask_data.shape}, mito={mito_data.shape}")
            return
        
        # 过滤数据并生成mito和mask
        mito_included, mito_excluded = filter_mito_by_mask_id(
            mask_data, mito_data, args.mask_ids, 
            output_mito_included=args.output_mito_included,
            output_mito_excluded=args.output_mito_excluded,
            output_mask_included=args.output_mask_included,
            output_mask_excluded=args.output_mask_excluded
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

