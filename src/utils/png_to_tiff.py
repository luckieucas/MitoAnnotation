#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm

def png_to_tiff(input_dir, output_path, compression="zlib", imagej=True):
    """
    将目录下所有PNG文件合并成一个3D TIFF文件
    
    参数:
        input_dir: 输入PNG文件所在目录
        output_path: 输出TIFF文件路径
        compression: 压缩方式，默认zlib
        imagej: 是否保存为ImageJ格式，默认True
    """
    input_path = Path(input_dir)
    
    # 获取所有PNG文件并排序
    png_files = sorted(input_path.glob("*.png"))
    
    if len(png_files) == 0:
        raise ValueError(f"在目录 {input_dir} 中未找到PNG文件")
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 读取第一张图片获取尺寸和数据类型
    first_img = np.array(Image.open(png_files[0]))
    height, width = first_img.shape[:2]
    dtype = first_img.dtype
    
    # 检查是否为灰度图或彩色图
    if first_img.ndim == 2:
        # 灰度图
        vol_shape = (len(png_files), height, width)
        is_color = False
    elif first_img.ndim == 3:
        # 彩色图
        channels = first_img.shape[2]
        vol_shape = (len(png_files), height, width, channels)
        is_color = True
    else:
        raise ValueError(f"不支持的图像维度: {first_img.ndim}")
    
    print(f"图像尺寸: {height}x{width}, 数据类型: {dtype}")
    if is_color:
        print(f"彩色图像，通道数: {vol_shape[3]}")
    else:
        print("灰度图像")
    
    # 预分配数组
    volume = np.empty(vol_shape, dtype=dtype)
    
    # 逐张读取PNG文件
    print("读取PNG文件...")
    for i, png_file in enumerate(tqdm(png_files)):
        img = np.array(Image.open(png_file))
        
        # 检查尺寸是否一致
        if img.shape[:2] != (height, width):
            raise ValueError(f"文件 {png_file.name} 的尺寸 {img.shape[:2]} 与第一张图片 ({height}, {width}) 不一致")
        
        volume[i] = img
    
    # 保存为TIFF
    print(f"保存为TIFF文件: {output_path}")
    
    # 设置元数据
    if is_color:
        metadata = {"axes": "ZYXC"}
        photometric = "rgb"
    else:
        metadata = {"axes": "ZYX"}
        photometric = "minisblack"
    
    tiff.imwrite(
        output_path,
        volume,
        imagej=imagej,
        photometric=photometric,
        metadata=metadata,
        bigtiff=True,
        compression=None if (compression is None or str(compression).lower() == "none") else compression
    )
    
    print(f"完成! 输出文件: {output_path}")
    print(f"最终数据形状: {volume.shape}")
    
    return volume.shape

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="将目录下所有PNG文件合并成一个3D TIFF文件（使用zlib压缩）")
    ap.add_argument("--input", required=True, help="输入PNG文件所在目录")
    ap.add_argument("--output", required=True, help="输出TIFF文件路径")
    ap.add_argument("--compression", type=str, default="zlib", help="压缩方式：zlib/lzw/none，默认zlib")
    ap.add_argument("--no_imagej", action="store_true", help="不使用ImageJ格式")
    
    args = ap.parse_args()
    
    shape = png_to_tiff(
        args.input,
        args.output,
        compression=args.compression,
        imagej=not args.no_imagej
    )
    print(f"成功保存! 形状: {shape}")

