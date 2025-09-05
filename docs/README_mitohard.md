# MitoHard 环境合并说明

## 概述
我已经成功将三个conda环境（empanada_env、sam、sam2_mito）合并到一个新的名为`mitohard`的环境中。

## 创建的文件
1. `mitohard.yml` - 原始合并版本
2. `mitohard_simple.yml` - 简化版本
3. `mitohard_final.yml` - 最终版本
4. `mitohard_compatible.yml` - 兼容版本（推荐使用）

## 环境特点
- **Python版本**: 3.12.9
- **PyTorch版本**: 2.5.1 (CUDA 12.1)
- **主要功能**: 包含所有三个原始环境的核心功能

### 核心组件
- **深度学习**: PyTorch, Lightning, MONAI, nnU-Net
- **图像分割**: SAM, SAM2, Micro-SAM, Mobile-SAM
- **医学影像**: Nibabel, PyDICOM, SimpleITK, Connected Components 3D
- **图像处理**: OpenCV, Albumentations, Kornia, Mahotas
- **可视化**: Napari, Empanada-Napari, Vispy, VTK
- **Jupyter**: JupyterLab, Notebook, IPython
- **数据格式**: HDF5, Zarr, TIFF, DICOM, NIfTI

## 使用方法

### 创建环境
```bash
# 使用兼容版本（推荐）
conda env create -f mitohard_compatible.yml

# 或者使用其他版本
conda env create -f mitohard_final.yml
conda env create -f mitohard_simple.yml
```

### 激活环境
```bash
conda activate mitohard
```

### 验证安装
```bash
# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 检查主要包
python -c "import napari, monai, segment_anything, sam2"
```

## 版本冲突处理
在合并过程中，我处理了以下版本冲突：
- 统一使用Python 3.12.9
- 统一使用PyTorch 2.5.1
- 统一使用OpenCV 4.11.0
- 统一使用JupyterLab 4.3.6
- 统一使用Napari 0.6.1

## 注意事项
1. 环境文件较大，安装可能需要较长时间
2. 如果遇到依赖冲突，可以尝试使用`--force-reinstall`选项
3. 建议在创建环境前先备份现有环境
4. 某些包可能需要额外的系统依赖

## 故障排除
如果遇到问题，可以尝试：
1. 更新conda: `conda update conda`
2. 清理缓存: `conda clean --all`
3. 使用mamba: `mamba env create -f mitohard_compatible.yml`

## 环境大小
预计环境大小约为8-12GB，包含所有依赖项。
