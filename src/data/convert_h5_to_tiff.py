import os
import h5py
import tifffile as tiff
import numpy as np
import argparse

def convert_h5_to_tif(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        base_name = os.path.splitext(filename)[0]
        tif_path = os.path.join(output_folder, base_name + '.tiff')
        if os.path.exists(tif_path):
            print(f"Skipping {filename} because {tif_path} already exists")
            continue
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            h5_path = os.path.join(input_folder, filename)
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
                    print(f"{base_name} has no data")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .h5/.hdf5 files to .tif")
    parser.add_argument("--input", type=str, required=True, help="Path to input folder containing .h5 files")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder for .tif files")

    args = parser.parse_args()
    convert_h5_to_tif(args.input, args.output)
