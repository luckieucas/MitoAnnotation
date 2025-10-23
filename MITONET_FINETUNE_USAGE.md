# MitoNet Fine-tune & Predict Usage Guide

这个脚本支持两种模式：
1. **训练模式（train）**：使用nnUNet数据集微调MitoNet模型
2. **预测模式（predict）**：使用训练好的模型对3D TIFF文件进行预测

## 1. 训练模式 (Train Mode)

### 基本用法

```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    /path/to/Dataset \
    ./checkpoints/my_model \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MyMitoNet \
    --iterations 1000
```

### 完整参数

```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    /path/to/Dataset004_MitoHardCardiac \
    ./checkpoints/cardiac_model \
    --mode train \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MitoNet_Cardiac \
    --iterations 1000 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --patch_size 256 \
    --finetune_layer all \
    --output_data_path ./data/cardiac_2d_slices
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset_path` | 必需 | nnUNet数据集路径 |
| `model_dir` | 必需 | 模型保存目录 |
| `--mode` | train | 模式选择：train或predict |
| `--model_config` | MitoNet_v1.yaml | 基础模型配置文件 |
| `--model_name` | FinetunedMitoNet | 新模型名称 |
| `--iterations` | 1000 | 训练迭代次数 |
| `--batch_size` | 16 | 批次大小 |
| `--learning_rate` | 0.003 | 最大学习率 |
| `--patch_size` | 256 | 训练patch大小 |
| `--finetune_layer` | all | 微调层级 |
| `--skip_conversion` | False | 跳过数据转换 |
| `--output_data_path` | auto | 2D切片保存路径 |
| `--test_after_training` | False | **训练后自动测试和评估** ⭐ |
| `--test_axes` | xy xz yz | 测试时使用的推理平面 |

### 💡 训练后自动测试功能 (NEW!)

使用 `--test_after_training` 参数可以在训练完成后自动：
1. 对 `imagesTs` 目录下的所有测试图像进行预测
2. 将预测结果保存到 `{dataset_path}/imagesTs_mitoNet_FT/` 目录
3. 与 `instancesTs` 中的ground truth进行对比评估
4. 生成详细的评估报告

**预测文件命名规则**：
- 去掉原始文件名中的 `_0000` 后缀
- 最终预测文件：`{filename}.tiff`（例如：`jrc_jurkat-1_recon-1_test1.tiff`）
- 单平面预测：`{filename}_xy.tiff`、`{filename}_xz.tiff`、`{filename}_yz.tiff`

**输出目录结构**：
```
Dataset003_MitoHardJurkat/
├── imagesTs/                           # 原始测试图像
├── instancesTs/                        # Ground truth
├── imagesTs_mitoNet_FT/                # 预测结果目录 ⭐
│   ├── jrc_jurkat-1_recon-1_test1.tiff         # 最终预测（共识）
│   ├── jrc_jurkat-1_recon-1_test1_xy.tiff      # xy平面预测
│   ├── jrc_jurkat-1_recon-1_test1_xz.tiff      # xz平面预测
│   ├── jrc_jurkat-1_recon-1_test1_yz.tiff      # yz平面预测
│   └── evaluation/                              # 评估结果
│       ├── evaluation_results.csv               # 详细结果
│       └── evaluation_results_summary.txt       # 汇总统计
```

### 训练示例

#### Dataset003_MitoHardJurkat (不带自动测试)

```bash
cd /projects/weilab/liupeng/MitoAnnotation

conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat \
    checkpoints/Dataset003_MitoHardJurkat \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MitoNet_Jurkat \
    --iterations 1000 \
    --batch_size 8
```

#### Dataset003_MitoHardJurkat (带自动测试和评估) ⭐ 推荐

```bash
cd /projects/weilab/liupeng/MitoAnnotation

conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat \
    checkpoints/Dataset003_MitoHardJurkat \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MitoNet_Jurkat \
    --iterations 1000 \
    --batch_size 8 \
    --test_after_training \
    --test_axes xy xz yz
```

训练完成后会自动：
- 在 `Dataset003_MitoHardJurkat/imagesTs_mitoNet_FT/` 生成预测结果
- 在 `Dataset003_MitoHardJurkat/imagesTs_mitoNet_FT/evaluation/` 生成评估报告

#### Dataset004_MitoHardCardiac (带自动测试)

```bash
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset004_MitoHardCardiac \
    checkpoints/Dataset004_MitoHardCardiac \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MitoNet_Cardiac \
    --iterations 2000 \
    --batch_size 16 \
    --test_after_training
```

预测结果将保存在：
- `Dataset004_MitoHardCardiac/imagesTs_mitoNet_FT/`

## 2. 预测模式 (Predict Mode)

### 基本用法

```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model /path/to/trained_model.yaml \
    --input /path/to/input.tiff \
    --output /path/to/output_dir \
    --use_gpu
```

### 完整参数

```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/cardiac_model/MitoNet_Cardiac.yaml \
    --input /path/to/test_volume.tiff \
    --output ./predictions/cardiac \
    --use_gpu \
    --axes xy xz yz \
    --confidence_thr 0.5 \
    --center_confidence_thr 0.1 \
    --min_size 500 \
    --min_extent 5 \
    --pixel_vote_thr 2
```

### 预测参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | train | 设置为 predict |
| `--trained_model` | 必需 | 训练好的模型配置文件 (.yaml) |
| `--input` | 必需 | 输入3D TIFF文件或目录 |
| `--output` | 必需 | 输出目录 |
| `--use_gpu` | False | 使用GPU |
| `--axes` | xy xz yz | 推理平面 |
| `--downsampling` | 1 | 下采样因子 |
| `--confidence_thr` | 0.5 | 分割置信度阈值 |
| `--center_confidence_thr` | 0.1 | 中心置信度阈值 |
| `--min_distance_object_centers` | 3 | 对象中心最小距离 |
| `--min_size` | 500 | 对象最小体素数 |
| `--min_extent` | 5 | 对象最小边界框范围 |
| `--pixel_vote_thr` | 2 | 正交平面共识的体素投票阈值 |

### 预测示例

#### 预测单个文件

```bash
cd /projects/weilab/liupeng/MitoAnnotation

conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/Dataset003_MitoHardJurkat/MitoNet_Jurkat.yaml \
    --input /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/imagesTs/jrc_jurkat-1_recon-1_test1_0000.tiff \
    --output predictions/jurkat_test1 \
    --use_gpu \
    --axes xy
```

#### 预测整个目录

```bash
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/Dataset004_MitoHardCardiac/MitoNet_Cardiac.yaml \
    --input /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset004_MitoHardCardiac/imagesTs \
    --output predictions/cardiac_all_tests \
    --use_gpu \
    --axes xy xz yz \
    --confidence_thr 0.6 \
    --min_size 1000
```

#### 只使用xy平面（快速预测）

```bash
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/my_model/MyMitoNet.yaml \
    --input /path/to/test.tiff \
    --output predictions/quick_test \
    --use_gpu \
    --axes xy
```

### 输出文件

预测模式会生成以下文件（自动去掉`_0000`后缀）：

- `{filename}.tiff` - **最终共识分割结果**（推荐使用）⭐
- `{filename}_xy.tiff` - xy平面的分割结果
- `{filename}_xz.tiff` - xz平面的分割结果（如果选择）
- `{filename}_yz.tiff` - yz平面的分割结果（如果选择）

例如，对于输入文件 `jrc_jurkat-1_recon-1_test1_0000.tiff`，输出：
- `jrc_jurkat-1_recon-1_test1.tiff` (主预测结果)
- `jrc_jurkat-1_recon-1_test1_xy.tiff`
- `jrc_jurkat-1_recon-1_test1_xz.tiff`
- `jrc_jurkat-1_recon-1_test1_yz.tiff`

## 3. 完整工作流程示例

### 步骤1：训练模型

```bash
cd /projects/weilab/liupeng/MitoAnnotation

# 训练Jurkat数据集
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat \
    checkpoints/jurkat_model \
    --model_config configs/MitoNet_v1.yaml \
    --model_name MitoNet_Jurkat \
    --iterations 1000 \
    --batch_size 8 \
    --learning_rate 0.001
```

训练完成后，模型会保存在：
- `checkpoints/jurkat_model/MitoNet_Jurkat.yaml` - 模型配置
- `checkpoints/jurkat_model/MitoNet_Jurkat.pth` - 模型权重

### 步骤2：使用训练好的模型进行预测

```bash
# 预测测试集
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/jurkat_model/MitoNet_Jurkat.yaml \
    --input /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/imagesTs \
    --output predictions/jurkat_test \
    --use_gpu \
    --axes xy xz yz
```

预测结果会保存在：
- `predictions/jurkat_test/jrc_jurkat-1_recon-1_test1_0000_consensus.tif`

## 4. 高级技巧

### 调整预测参数以提高质量

对于小对象（例如小线粒体）：
```bash
--min_size 200 \
--min_extent 3 \
--confidence_thr 0.4
```

对于大对象：
```bash
--min_size 1000 \
--min_extent 10 \
--confidence_thr 0.6
```

### 只使用单个平面（快速预测）

如果只关心xy平面的结果：
```bash
--axes xy
```

### 批量处理脚本

```bash
#!/bin/bash
# batch_predict.sh

MODEL=checkpoints/my_model/MyModel.yaml
INPUT_DIR=/path/to/test/images
OUTPUT_BASE=predictions

for file in $INPUT_DIR/*.tiff; do
    filename=$(basename "$file" .tiff)
    echo "Processing $filename..."
    
    conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
        --mode predict \
        --trained_model $MODEL \
        --input "$file" \
        --output "$OUTPUT_BASE/$filename" \
        --use_gpu \
        --axes xy
done

echo "All predictions completed!"
```

## 5. 常见问题

### Q: GPU内存不足怎么办？
A: 尝试以下方法：
- 训练时：减小 `--batch_size` (例如改为4或2)
- 预测时：增大 `--downsampling` (例如设置为2)
- 只使用单个平面：`--axes xy`

### Q: 训练后找不到模型文件？
A: 模型保存在 `{model_dir}/{model_name}.yaml` 和 `{model_dir}/{model_name}.pth`

### Q: 预测太慢？
A: 
- 使用 `--axes xy` 只在一个平面预测
- 增加 `--downsampling` 参数
- 确保使用了 `--use_gpu`

### Q: 预测结果不理想？
A: 尝试调整：
- `--confidence_thr`: 降低以获得更多对象
- `--min_size`: 调整以过滤小对象
- `--pixel_vote_thr`: 调整共识阈值

## 6. 性能对比

| 配置 | 速度 | 质量 | 建议使用场景 |
|------|------|------|-------------|
| xy only | ⚡⚡⚡ | ⭐⭐ | 快速测试 |
| xy + xz | ⚡⚡ | ⭐⭐⭐ | 平衡模式 |
| xy + xz + yz | ⚡ | ⭐⭐⭐⭐ | 最佳质量 |

## 7. 故障排除

### 错误：No module named 'empanada'
```bash
# 确保在正确的conda环境中
conda activate mitohard
# 或使用
conda run -n mitohard python ...
```

### 错误：CUDA out of memory
```bash
# 训练时减小batch size
--batch_size 4

# 预测时增加downsampling
--downsampling 2
```

### 错误：找不到训练好的模型
```bash
# 检查模型路径是否正确
ls checkpoints/my_model/MyModel.yaml
```

## 8. 脚本对比

| 功能 | mitoNet_finetune_from_nnunet.py | MitoNet_baseline.py |
|------|--------------------------------|---------------------|
| 训练 | ✅ | ❌ |
| 预测 | ✅ | ✅ |
| nnUNet数据 | ✅ | ❌ |
| 自动转换 | ✅ | ❌ |
| 使用预训练模型 | ✅ | ✅ |
| 使用fine-tuned模型 | ✅ | ✅ |

新的`mitoNet_finetune_from_nnunet.py`是一个all-in-one工具，包含了训练和预测两个功能！

