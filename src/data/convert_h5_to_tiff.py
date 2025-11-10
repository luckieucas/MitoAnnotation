import os
import h5py
import tifffile as tiff
import numpy as np
import argparse

def convert_single_h5_to_tif(h5_path, tif_path):
    """
    转换单个h5文件到tif文件
    
    Args:
        h5_path: 输入的h5文件路径
        tif_path: 输出的tif文件路径
    """
    filename = os.path.basename(h5_path)
    
    if os.path.exists(tif_path):
        print(f"Skipping {filename} because {tif_path} already exists")
        return
    
    with h5py.File(h5_path, 'r') as f:
        print(f"Reading {filename}, available datasets: {list(f.keys())}")

        # 尝试读取 'data' 或第一个 dataset
        if 'data' in f:
            data = f['data'][()]
        else:
            first_key = list(f.keys())[0]
            data = f[first_key][()]
            print(f"Using dataset '{first_key}'")

        # 生成 tif 文件名并保存
        if data.max() == 0:
            print(f"{filename} has no data")
#                    continue
        # Normalize data
        # normalize image
        print(f"data type:{data.dtype}")
        if filename.endswith('im.h5'):
             print(f"data shape: {data.shape},max value: {np.max(data)}")
             data = (data - data.min()) / (data.max() - data.min())
             data = data*255
             data = data.astype(np.uint8)
             tiff.imwrite(tif_path, data, compression="zlib")  # 根据需要可改为 np.float32
        else:
            tiff.imwrite(tif_path, data.astype(np.uint16), compression="zlib")
        print(f"Saved {tif_path}")

def convert_h5_to_tif(input_path, output_path):
    """
    转换h5文件到tif文件，支持单个文件或文件夹
    
    Args:
        input_path: 输入路径，可以是单个h5文件或包含h5文件的文件夹
        output_path: 输出路径，可以是单个tif文件或输出文件夹
    """
    # 判断输入是文件还是文件夹
    if os.path.isfile(input_path):
        # 单个文件处理
        if not (input_path.endswith('.h5') or input_path.endswith('.hdf5')):
            print(f"Error: {input_path} is not a .h5 or .hdf5 file")
            return
        
        # 如果output_path是文件夹，则生成输出文件名
        if os.path.isdir(output_path):
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            tif_path = os.path.join(output_path, base_name + '.tiff')
        else:
            # output_path是文件路径
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            tif_path = output_path
        
        convert_single_h5_to_tif(input_path, tif_path)
        
    elif os.path.isdir(input_path):
        # 文件夹处理
        os.makedirs(output_path, exist_ok=True)
        
        for filename in os.listdir(input_path):
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                h5_path = os.path.join(input_path, filename)
                base_name = os.path.splitext(filename)[0]
                tif_path = os.path.join(output_path, base_name + '.tiff')
                convert_single_h5_to_tif(h5_path, tif_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .h5/.hdf5 files to .tif, supports both single file and folder")
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5 file or folder containing .h5 files")
    parser.add_argument("--output", type=str, required=True, help="Path to output .tif file or folder for .tif files")

    args = parser.parse_args()
    convert_h5_to_tif(args.input, args.output)
