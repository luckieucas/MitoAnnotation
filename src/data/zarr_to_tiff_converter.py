#!/usr/bin/env python3
"""
Zarr to TIFF Converter

This script reads zarr files and converts them to compressed TIFF files.
The zarr files are expected to have 4 channels:
- First 3 channels: affinity maps
- Last channel: skeleton distance transform (-1 to 0 is background, 0 to 1 is foreground)

Author: Generated for MitoAnnotation project
Date: 2025
"""

import os
import zarr
import numpy as np
import tifffile
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import disk, ball
import multiprocessing as mp
from functools import partial
import time
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    from skimage.feature import peak_local_max as peak_local_maxima

# 设置日志
def setup_logging(log_level=logging.INFO, log_file='zarr_conversion.log'):
    """
    设置多进程安全的日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

# 初始化日志
logger = setup_logging()

def detect_data_format(zarr_data):
    """
    自动检测zarr数据的格式
    
    Args:
        zarr_data: zarr数据对象
    
    Returns:
        tuple: (format_type, channels_dim, z_dim, y_dim, x_dim)
    """
    shape = zarr_data.shape
    logger.info(f"原始数据形状: {shape}")
    
    if len(shape) == 4:
        # 可能的格式: (C, Z, Y, X) 或 (Z, C, Y, X)
        if shape[0] <= 10:  # 第一维是通道数
            logger.info("检测到格式: (C, Z, Y, X)")
            return "CZYX", 0, 1, 2, 3
        else:  # 第一维是Z轴
            logger.info("检测到格式: (Z, C, Y, X)")
            return "ZCYX", 1, 0, 2, 3
    else:
        logger.warning(f"不支持的数据维度: {len(shape)}D")
        return None, None, None, None, None

def apply_watershed_segmentation(skeleton_data, threshold=0.0, min_distance=5, min_size=100):
    """
    对skeleton distance数据进行watershed分割
    
    Args:
        skeleton_data: skeleton distance数据 (范围[-1, 1])
        threshold: 阈值，大于此值认为是前景
        min_distance: 局部最大值之间的最小距离
        min_size: 分割区域的最小大小
    
    Returns:
        tuple: (binary_mask, watershed_result)
    """
    # 创建二值掩码
    binary_mask = skeleton_data > threshold
    logger.info(f"二值掩码统计: 前景像素 {np.sum(binary_mask)}, 背景像素 {np.sum(~binary_mask)}")
    
    # 形态学操作清理二值掩码
    binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # 根据数据维度选择合适的结构元素
    if len(binary_mask.shape) == 3:
        # 3D数据使用ball结构元素
        structure = ball(1)
    else:
        # 2D数据使用disk结构元素
        structure = disk(1)
    
    binary_mask = ndimage.binary_opening(binary_mask, structure=structure)
    
    # 计算距离变换
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # 找到局部最大值作为种子点
    local_maxima = peak_local_maxima(distance, min_distance=min_distance, threshold_abs=0.1)
    
    # 创建标记图像
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (z, y, x) in enumerate(local_maxima):
        markers[z, y, x] = i + 1
    
    # 应用watershed分割
    watershed_result = watershed(-distance, markers, mask=binary_mask)
    
    # 移除过小的区域
    unique_labels, counts = np.unique(watershed_result, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if count < min_size and label > 0:  # 保留背景标签0
            watershed_result[watershed_result == label] = 0
    
    # 重新标记连续的区域
    watershed_result = ndimage.label(watershed_result)[0]
    
    logger.info(f"Watershed分割完成: 找到 {np.max(watershed_result)} 个区域")
    
    return binary_mask, watershed_result

def read_zarr_and_save_tiff(zarr_path, output_dir, compression='lzw', threshold=0.0, 
                           min_distance=5, min_size=100, apply_watershed=True):
    """
    读取zarr文件并将各个通道保存为压缩的TIFF文件
    
    Args:
        zarr_path: zarr文件路径
        output_dir: 输出目录
        compression: TIFF压缩方式 ('lzw', 'deflate', 'jpeg'等)
        threshold: skeleton distance的阈值
        min_distance: watershed分割中局部最大值之间的最小距离
        min_size: 分割区域的最小大小
        apply_watershed: 是否应用watershed分割
    """
    try:
        logger.info(f"开始处理: {zarr_path}")
        
        # 打开zarr文件
        zarr_data = zarr.open(zarr_path, mode='r')
        
        # 检测数据格式
        format_type, ch_dim, z_dim, y_dim, x_dim = detect_data_format(zarr_data)
        
        if format_type is None:
            logger.error(f"无法识别数据格式: {zarr_path}")
            return
        
        # 读取所有数据
        data = zarr_data[:]
        logger.info(f"数据形状: {data.shape}")
        
        # 根据格式重新排列数据
        if format_type == "ZCYX":
            # 原始格式: (Z, C, Y, X) -> 需要转置为 (C, Z, Y, X)
            data = np.transpose(data, (1, 0, 2, 3))
            logger.info(f"转置后数据形状: {data.shape}")
        
        # 检查通道数
        num_channels = data.shape[0]
        logger.info(f"通道数: {num_channels}")
        
        if num_channels < 4:
            logger.warning(f"通道数不足4个，只有{num_channels}个通道")
            return
        
        # 创建输出目录
        zarr_name = Path(zarr_path).stem
        zarr_output_dir = Path(output_dir) / zarr_name
        zarr_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存前3个通道 (affinity)
        for i in range(min(3, num_channels)):
            channel_data = data[i]
            print(f"affinity max: {np.max(channel_data)}, min: {np.min(channel_data)}")
            output_path = zarr_output_dir / f"affinity_ch{i+1}.tif"
            
            # 统一转换为float32格式
            channel_data = channel_data.astype(np.float32)
            
            # 保存为压缩的TIFF
            tifffile.imwrite(
                output_path,
                channel_data,
                compression=compression,
                compressionargs={'level': 6} if compression == 'deflate' else None
            )
            logger.info(f"已保存: {output_path}, 形状: {channel_data.shape}, 数据类型: {channel_data.dtype}")
        
        # 保存最后一个通道 (skeleton distance transform)
        if num_channels >= 4:
            skeleton_data = data[-1]
            print(f"skeleton max: {np.max(skeleton_data)}, min: {np.min(skeleton_data)}")
            skeleton_output_path = zarr_output_dir / "skeleton_distance.tif"
            
            # 统一转换为float32格式
            skeleton_data = skeleton_data.astype(np.float32)
            
            # 保存原始skeleton distance数据
            tifffile.imwrite(
                skeleton_output_path,
                skeleton_data,
                compression=compression,
                compressionargs={'level': 6} if compression == 'deflate' else None
            )
            logger.info(f"已保存: {skeleton_output_path}, 形状: {skeleton_data.shape}, 数据类型: {skeleton_data.dtype}")
            
            # 如果启用watershed分割
            if apply_watershed:
                logger.info(f"开始watershed分割，阈值: {threshold}")
                
                # 应用watershed分割
                binary_mask, watershed_result = apply_watershed_segmentation(
                    skeleton_data, threshold, min_distance, min_size
                )
                
                # 保存二值掩码
                binary_mask_path = zarr_output_dir / "binary_mask.tif"
                binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
                tifffile.imwrite(
                    binary_mask_path,
                    binary_mask_uint8,
                    compression=compression,
                    compressionargs={'level': 6} if compression == 'deflate' else None
                )
                logger.info(f"已保存: {binary_mask_path}, 形状: {binary_mask_uint8.shape}, 数据类型: {binary_mask_uint8.dtype}")
                
                # 保存watershed分割结果
                watershed_path = zarr_output_dir / "watershed_segmentation.tif"
                watershed_result_uint16 = watershed_result.astype(np.uint16)
                tifffile.imwrite(
                    watershed_path,
                    watershed_result_uint16,
                    compression=compression,
                    compressionargs={'level': 6} if compression == 'deflate' else None
                )
                logger.info(f"已保存: {watershed_path}, 形状: {watershed_result_uint16.shape}, 数据类型: {watershed_result_uint16.dtype}")
        
        logger.info(f"完成处理: {zarr_path}")
        
    except Exception as e:
        logger.error(f"处理 {zarr_path} 时出错: {e}")

def process_single_zarr(args):
    """
    处理单个zarr文件的包装函数，用于多进程处理
    
    Args:
        args: 包含所有参数的元组
    """
    zarr_file, output_folder, compression, threshold, min_distance, min_size, apply_watershed = args
    
    # 为每个进程创建独立的日志记录器
    process_logger = logging.getLogger(f"Process-{os.getpid()}")
    process_logger.setLevel(logging.INFO)
    
    try:
        process_logger.info(f"进程 {os.getpid()} 开始处理: {zarr_file}")
        read_zarr_and_save_tiff(str(zarr_file), str(output_folder), compression, 
                               threshold, min_distance, min_size, apply_watershed)
        process_logger.info(f"进程 {os.getpid()} 完成处理: {zarr_file}")
        return True, str(zarr_file)
    except Exception as e:
        process_logger.error(f"进程 {os.getpid()} 处理 {zarr_file} 时出错: {e}")
        return False, str(zarr_file)

def process_zarr_folder(input_folder, output_folder, compression='lzw', threshold=0.0, 
                       min_distance=5, min_size=100, apply_watershed=True, num_processes=None):
    """
    处理文件夹中的所有zarr文件（支持并行处理）
    
    Args:
        input_folder: 包含zarr文件的输入文件夹
        output_folder: 输出文件夹
        compression: TIFF压缩方式
        threshold: skeleton distance的阈值
        min_distance: watershed分割中局部最大值之间的最小距离
        min_size: 分割区域的最小大小
        apply_watershed: 是否应用watershed分割
        num_processes: 并行进程数，None表示使用CPU核心数
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有zarr文件
    zarr_files = list(input_path.glob("*.zarr"))
    
    if not zarr_files:
        logger.warning(f"在 {input_folder} 中没有找到zarr文件")
        return
    
    logger.info(f"找到 {len(zarr_files)} 个zarr文件")
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(zarr_files))
    
    logger.info(f"使用 {num_processes} 个进程进行并行处理")
    
    # 准备参数
    process_args = [(zarr_file, output_path, compression, threshold, min_distance, min_size, apply_watershed) 
                   for zarr_file in zarr_files]
    
    # 如果只有一个文件或进程数设为1，则串行处理
    if num_processes == 1 or len(zarr_files) == 1:
        logger.info("使用串行处理模式")
        for args in tqdm(process_args, desc="处理zarr文件"):
            process_single_zarr(args)
    else:
        # 并行处理
        logger.info("使用并行处理模式")
        start_time = time.time()
        
        with mp.Pool(processes=num_processes) as pool:
            # 使用imap_unordered来支持进度条
            results = list(tqdm(
                pool.imap_unordered(process_single_zarr, process_args),
                total=len(process_args),
                desc="处理zarr文件"
            ))
        
        end_time = time.time()
        
        # 统计结果
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        logger.info(f"并行处理完成: 成功 {successful} 个，失败 {failed} 个")
        logger.info(f"总耗时: {end_time - start_time:.2f} 秒")
        
        # 报告失败的文件
        if failed > 0:
            failed_files = [filename for success, filename in results if not success]
            logger.warning(f"失败的文件: {failed_files}")

def validate_zarr_file(zarr_path):
    """
    验证zarr文件的有效性
    
    Args:
        zarr_path: zarr文件路径
    
    Returns:
        bool: 文件是否有效
    """
    try:
        zarr_data = zarr.open(zarr_path, mode='r')
        
        # 检查数据维度
        if len(zarr_data.shape) != 4:
            logger.error(f"数据维度错误: 期望4D，实际{len(zarr_data.shape)}D")
            return False
        
        # 检查通道数
        shape = zarr_data.shape
        if shape[0] <= 10:  # (C, Z, Y, X) 格式
            if shape[0] < 4:
                logger.error(f"通道数错误: 期望至少4个通道，实际{shape[0]}个通道")
                return False
        elif shape[1] <= 10:  # (Z, C, Y, X) 格式
            if shape[1] < 4:
                logger.error(f"通道数错误: 期望至少4个通道，实际{shape[1]}个通道")
                return False
        else:
            logger.error(f"通道数错误: 无法确定通道维度，形状{shape}")
            return False
        
        logger.info(f"文件验证通过: {zarr_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件验证失败 {zarr_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='将zarr文件转换为压缩的TIFF文件')
    parser.add_argument('input_folder', help='包含zarr文件的输入文件夹路径')
    parser.add_argument('output_folder', help='输出TIFF文件的文件夹路径')
    parser.add_argument('--compression', '-c', default='zlib', 
                       choices=['lzw', 'deflate', 'jpeg', 'none'],
                       help='TIFF压缩方式 (默认: lzw)')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='在处理前验证zarr文件')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别')
    parser.add_argument('--threshold', '-t', type=float, default=0.0,
                       help='skeleton distance的阈值 (默认: 0.0)')
    parser.add_argument('--min-distance', type=int, default=5,
                       help='watershed分割中局部最大值之间的最小距离 (默认: 5)')
    parser.add_argument('--min-size', type=int, default=100,
                       help='分割区域的最小大小 (默认: 100)')
    parser.add_argument('--no-watershed', action='store_true',
                       help='禁用watershed分割，只保存原始数据')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='并行进程数，默认为CPU核心数')
    parser.add_argument('--serial', action='store_true',
                       help='强制使用串行处理模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    setup_logging(getattr(logging, args.log_level))
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input_folder):
        logger.error(f"错误: 输入文件夹 {args.input_folder} 不存在")
        return
    
    # 确定进程数
    if args.serial:
        num_processes = 1
    else:
        num_processes = args.processes
    
    logger.info(f"输入文件夹: {args.input_folder}")
    logger.info(f"输出文件夹: {args.output_folder}")
    logger.info(f"压缩方式: {args.compression}")
    logger.info(f"阈值: {args.threshold}")
    logger.info(f"最小距离: {args.min_distance}")
    logger.info(f"最小区域大小: {args.min_size}")
    logger.info(f"启用watershed: {not args.no_watershed}")
    logger.info(f"并行进程数: {num_processes if num_processes else '自动检测'}")
    
    # 如果启用验证，先验证所有zarr文件
    if args.validate:
        logger.info("开始验证zarr文件...")
        input_path = Path(args.input_folder)
        zarr_files = list(input_path.glob("*.zarr"))
        
        valid_files = []
        for zarr_file in zarr_files:
            if validate_zarr_file(str(zarr_file)):
                valid_files.append(zarr_file)
        
        if not valid_files:
            logger.error("没有找到有效的zarr文件，退出")
            return
        
        logger.info(f"验证完成，找到 {len(valid_files)} 个有效文件")
    
    # 处理zarr文件
    process_zarr_folder(args.input_folder, args.output_folder, args.compression,
                       args.threshold, args.min_distance, args.min_size, not args.no_watershed, num_processes)

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    main()
