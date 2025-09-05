# Mito过滤脚本使用说明

这个目录包含了根据指定的mask id过滤mito数据的脚本。

## 文件说明

- `filter_mito_by_mask_id.py` - 主要的过滤脚本
- `example_filter_mito.py` - 使用示例脚本
- `test_filter_mito.py` - 测试脚本
- `README_FILTER_MITO.md` - 本说明文件

## 功能描述

这些脚本可以读取两个h5文件：
- `mitoEM-H_den_mask.h5` - 包含mask id信息
- `mitoEM-H_den_mito.h5` - 包含mito数据

然后根据指定的mask id列表，过滤掉不在有效mask区域内的mito数据。

## 安装依赖

确保安装了必要的Python包：

```bash
pip install h5py numpy
```

## 基本用法

### 1. 直接使用主脚本

```bash
python filter_mito_by_mask_id.py \
  --mask_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mask.h5 \
  --mito_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mito.h5 \
  --mask_ids 1 2 3 \
  --output filtered_mito.h5
```

### 2. 运行示例脚本

```bash
python example_filter_mito.py
```

### 3. 运行测试脚本

```bash
python test_filter_mito.py
```

## 参数说明

- `--mask_file`: mask h5文件路径
- `--mito_file`: mito h5文件路径
- `--mask_ids`: 有效的mask id列表（空格分隔）
- `--output`: 输出文件路径（可选，不指定则不保存）
- `--mask_dataset`: mask文件中的数据集键名（可选）
- `--mito_dataset`: mito文件中的数据集键名（可选）

## 使用示例

### 示例1：过滤并保存结果

```bash
python filter_mito_by_mask_id.py \
  --mask_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mask.h5 \
  --mito_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mito.h5 \
  --mask_ids 1 2 3 4 \
  --output filtered_mito_ids_1_2_3_4.h5
```

### 示例2：只查看过滤结果，不保存

```bash
python filter_mito_by_mask_id.py \
  --mask_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mask.h5 \
  --mito_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mito.h5 \
  --mask_ids 5 10 15
```

### 示例3：指定数据集键名

```bash
python filter_mito_by_mask_id.py \
  --mask_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mask.h5 \
  --mito_file /projects/weilab/dataset/MitoLE/wei20/mitoEM-H_den_mito.h5 \
  --mask_ids 1 2 3 \
  --mask_dataset "segmentation" \
  --mito_dataset "mitochondria" \
  --output filtered_mito.h5
```

## 输出说明

脚本会输出以下信息：
- 文件读取状态
- 数据形状和类型
- 过滤前后的统计信息
- 有效的mask id列表
- 无效的mask id列表
- 过滤掉的mito像素数量

## 注意事项

1. 确保mask文件和mito文件的数据形状一致
2. 如果文件很大，处理可能需要一些时间
3. 输出文件会使用gzip压缩来节省空间
4. 脚本会自动检测h5文件中的数据集键名

## 故障排除

### 常见错误

1. **文件不存在**: 检查文件路径是否正确
2. **数据形状不匹配**: 确保mask和mito文件来自同一数据集
3. **内存不足**: 对于大文件，可能需要增加系统内存

### 调试建议

1. 先运行测试脚本验证功能
2. 使用小数据集测试
3. 检查日志输出了解处理过程

## 技术支持

如果遇到问题，请检查：
1. Python版本（建议3.7+）
2. 依赖包版本
3. 文件格式和路径
4. 系统内存和磁盘空间

