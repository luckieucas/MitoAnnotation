#!/usr/bin/env python3
"""
Example script to run MitoNet baseline on nnUNet raw directory structure.
"""

import subprocess
import sys
import os

def main():
    # 配置参数
    input_path = "/path/to/nnUNet_raw"  # 替换为你的nnUNet raw目录路径
    datasets = ["Dataset001", "Dataset002", "Dataset003"]  # 替换为你要处理的数据集名称
    config_path = "configs/MitoNet_v1.yaml"  # 替换为你的配置文件路径
    
    # 构建命令
    cmd = [
        "python", "mitoNet_baseline.py",
        input_path,
        "--datasets"] + datasets + [
        "--config_path", config_path,
        "--use_gpu",  # 如果使用GPU
        "--evaluate",  # 如果需要进行评估
        "--confidence_thr", "0.5",
        "--center_confidence_thr", "0.1",
        "--min_size", "500",
        "--min_extent", "5"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "="*60)
        print("Processing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Python or mitoNet_baseline.py not found. Please check your environment.")
        sys.exit(1)

if __name__ == "__main__":
    main()
