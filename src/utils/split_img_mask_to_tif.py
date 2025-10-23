import argparse
import h5py
import os
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm

# 支持的文件扩展名
SUPPORTED_EXTS = {".h5", ".hdf5", ".tif", ".tiff"}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split a large 3D image and/or mask (H5 or TIFF) into fixed-size blocks and save them as TIFF or H5 files. "
                    "Input can be a single file or a folder containing files. Each file's results will be saved in a subfolder "
                    "named after the original file."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the input 3D image file (H5 or TIFF).")
    parser.add_argument("--mito", type=str, required=True, help="Path to the input 3D mitochondria instance mask file (H5 or TIFF). Used for filtering and will be saved as blocks.")
    parser.add_argument("--mask", type=str, help="Path to the input 3D mask file (H5 or TIFF). Optional.")
    parser.add_argument("--block_size", type=int, default=512, help="Block size for x and y dimensions. Default is 512.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the blocks.")
    parser.add_argument("--output_format", type=str, default="tiff", choices=["tiff", "h5"],
                        help="Output file format: 'tiff' or 'h5'. Default is 'tiff'.")
    parser.add_argument("--mito_size_threshold", type=int, default=1000, help="Minimum mitochondria size threshold. Default is 1000.")
    return parser.parse_args()

def load_data(file_path):
    """Load a 3D dataset from a file.
       For H5 files, the dataset is assumed to be stored under the key 'main'.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".h5", ".hdf5"]:
        with h5py.File(file_path, "r") as f:
            first_key = list(f.keys())[0]
            data = f[first_key][:]
        return data
    elif ext in [".tif", ".tiff"]:
        data = imread(file_path)
        return data
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def save_as_tiff(output_path, data):
    """Save the given data as a TIFF file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imwrite(output_path, data, compression="zlib")

def save_as_h5(output_path, data):
    """Save the given data as an H5 file with a dataset named 'block'."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("block", data=data, compression="gzip")

def check_mito_size(mito_block, threshold=1000):
    """
    检查mito block中的线粒体instance总大小是否达到阈值。
    mito_block: 3D instance mask，每个instance有不同的ID值
    threshold: 最小的线粒体像素数阈值
    返回: True 如果线粒体总大小 >= threshold，False 否则
    """
    # 统计所有非零像素（线粒体像素）的数量
    mito_size = np.count_nonzero(mito_block)
    return mito_size >= threshold

def split_and_save_single(data, block_size, output_dir, output_format, data_type, prefix, mito_data=None, mito_threshold=1000):
    """
    对单个文件中的3D数据进行分块保存，并将结果保存到 output_dir/prefix/ 下。
    data_type: "image" 或 "mask"，用于构造输出文件夹和文件名
    prefix: 原始文件名（不含扩展名），作为子文件夹名称
    mito_data: 可选的线粒体instance mask数据，用于过滤
    mito_threshold: 线粒体大小阈值
    """
    base_folder = os.path.join(output_dir, prefix)
    subfolder = os.path.join(base_folder, f"{data_type}s")
    os.makedirs(subfolder, exist_ok=True)
    
    z, x, y = data.shape
    x_blocks = (x + block_size - 1) // block_size
    y_blocks = (y + block_size - 1) // block_size
    ext = output_format
    
    skipped_count = 0

    for i in tqdm(range(x_blocks), desc=f"Processing {prefix} {data_type} x-blocks"):
        for j in tqdm(range(y_blocks), desc=f"Processing {prefix} {data_type} y-blocks", leave=False):
            x_start = i * block_size
            x_end = min((i + 1) * block_size, x)
            y_start = j * block_size
            y_end = min((j + 1) * block_size, y)

            block = data[:, x_start:x_end, y_start:y_end]
            
            # 如果提供了mito数据，检查该block的线粒体大小
            if mito_data is not None:
                mito_block = mito_data[:, x_start:x_end, y_start:y_end]
                if not check_mito_size(mito_block, mito_threshold):
                    skipped_count += 1
                    continue
            
            filename = f"{prefix}_{data_type}_x{x_start}_{x_end}_y{y_start}_{y_end}.{ext}"
            output_path = os.path.join(subfolder, filename)
            if output_format == "tiff":
                save_as_tiff(output_path, block)
            else:
                save_as_h5(output_path, block)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} blocks due to insufficient mitochondria size.")

def split_and_save_all(image_data, mito_data, mask_data, block_size, output_dir, output_format, prefix, mito_threshold=1000):
    """
    同时对 image、mito 和 mask（可选）进行分块保存，它们的形状必须相同。
    结果保存在 output_dir/prefix/ 下的 images、mitos 和 masks（如果提供）子文件夹中。
    prefix: 原始文件名（不含扩展名）
    image_data: 图像数据
    mito_data: 线粒体instance mask数据，用于过滤并保存
    mask_data: 可选的mask数据
    mito_threshold: 线粒体大小阈值
    """
    base_folder = os.path.join(output_dir, prefix)
    image_outdir = os.path.join(base_folder, "images")
    mito_outdir = os.path.join(base_folder, "mitos")
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(mito_outdir, exist_ok=True)
    
    # 如果提供了mask，创建masks文件夹
    if mask_data is not None:
        mask_outdir = os.path.join(base_folder, "masks")
        os.makedirs(mask_outdir, exist_ok=True)
    
    z, x, y = image_data.shape
    x_blocks = (x + block_size - 1) // block_size
    y_blocks = (y + block_size - 1) // block_size
    ext = output_format
    
    skipped_count = 0
    saved_count = 0

    for i in tqdm(range(x_blocks), desc=f"Processing {prefix} x-blocks"):
        for j in tqdm(range(y_blocks), desc=f"Processing {prefix} y-blocks", leave=False):
            x_start = i * block_size
            x_end = min((i + 1) * block_size, x)
            y_start = j * block_size
            y_end = min((j + 1) * block_size, y)

            # 检查该block的线粒体大小
            mito_block = mito_data[:, x_start:x_end, y_start:y_end]
            if not check_mito_size(mito_block, mito_threshold):
                skipped_count += 1
                continue

            # 提取所有需要的blocks
            image_block = image_data[:, x_start:x_end, y_start:y_end]

            # 构造文件名
            image_filename = f"{prefix}_x{x_start}_{x_end}_y{y_start}_{y_end}_0000.{ext}"
            mito_filename = f"{prefix}_x{x_start}_{x_end}_y{y_start}_{y_end}.{ext}"

            # 保存 image 和 mito blocks
            if output_format == "tiff":
                save_as_tiff(os.path.join(image_outdir, image_filename), image_block)
                save_as_tiff(os.path.join(mito_outdir, mito_filename), mito_block)
            else:
                save_as_h5(os.path.join(image_outdir, image_filename), image_block)
                save_as_h5(os.path.join(mito_outdir, mito_filename), mito_block)
            
            # 如果提供了mask，也保存mask block
            if mask_data is not None:
                mask_block = mask_data[:, x_start:x_end, y_start:y_end]
                mask_filename = f"{prefix}_x{x_start}_{x_end}_y{y_start}_{y_end}.{ext}"
                if output_format == "tiff":
                    save_as_tiff(os.path.join(mask_outdir, mask_filename), mask_block)
                else:
                    save_as_h5(os.path.join(mask_outdir, mask_filename), mask_block)
            
            saved_count += 1
    
    print(f"Saved {saved_count} blocks, skipped {skipped_count} blocks due to insufficient mitochondria size.")

def get_file_list(input_path):
    """
    如果 input_path 为文件夹，则返回其中所有支持的文件的完整路径列表；
    如果是单个文件，则返回列表中仅包含该文件路径。
    """
    if os.path.isdir(input_path):
        files = []
        for f in sorted(os.listdir(input_path)):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                files.append(os.path.join(input_path, f))
        return files
    elif os.path.isfile(input_path):
        return [input_path]
    else:
        raise ValueError(f"Path {input_path} is neither a file nor a directory.")

def main():
    args = parse_arguments()

    # 加载必需的 image 和 mito 数据
    if not os.path.isfile(args.image):
        raise ValueError(f"Image path {args.image} is not a valid file.")
    if not os.path.isfile(args.mito):
        raise ValueError(f"Mito path {args.mito} is not a valid file.")
    
    print(f"Loading image from {args.image}...")
    image_data = load_data(args.image)
    print(f"Image shape: {image_data.shape}")
    
    print(f"Loading mito mask from {args.mito}...")
    mito_data = load_data(args.mito)
    print(f"Mito shape: {mito_data.shape}")
    
    # 加载可选的 mask 数据
    mask_data = None
    if args.mask:
        if not os.path.isfile(args.mask):
            raise ValueError(f"Mask path {args.mask} is not a valid file.")
        print(f"Loading mask from {args.mask}...")
        mask_data = load_data(args.mask)
        print(f"Mask shape: {mask_data.shape}")
    
    # 检查形状是否一致
    if mask_data is not None and (image_data.shape != mask_data.shape or image_data.shape != mito_data.shape):
        raise ValueError(f"Shape mismatch: image {image_data.shape}, mito {mito_data.shape}, mask {mask_data.shape}")
    elif image_data.shape != mito_data.shape:
        raise ValueError(f"Shape mismatch: image {image_data.shape}, mito {mito_data.shape}")
    
    # 获取前缀名（使用image文件名）
    prefix = os.path.splitext(os.path.basename(args.image))[0]
    
    # 执行分块处理
    split_and_save_all(image_data, mito_data, mask_data, args.block_size, args.output_dir, 
                      args.output_format, prefix, mito_threshold=args.mito_size_threshold)

if __name__ == "__main__":
    main()
