# 更新总结：MitoNet Fine-tune 脚本功能增强

## 📅 更新日期
2025-10-16

## 🎯 主要更新

### 1. 训练和预测双模式支持
脚本现在支持两种工作模式：
- **训练模式（train）**：微调MitoNet模型
- **预测模式（predict）**：使用训练好的模型进行3D预测

### 2. 训练后自动测试和评估 ⭐
新增 `--test_after_training` 参数，可以在训练完成后自动：
- 对 `imagesTs` 目录下的所有测试图像进行预测
- 将预测结果保存到 `{dataset_path}/imagesTs_mitoNet_FT/` 目录
- 与 `instancesTs` 中的ground truth进行对比评估
- 生成详细的评估报告（CSV和文本汇总）

### 3. 改进的文件命名规则
- ✅ 自动去掉输入文件名中的 `_0000` 后缀
- ✅ 最终预测文件不再使用 `_consensus` 后缀
- ✅ 输出格式统一为 `.tiff`

### 4. 标准化输出目录结构
预测结果保存在与 `imagesTs` 并列的 `imagesTs_mitoNet_FT` 目录中，方便与其他方法对比：

```
Dataset003_MitoHardJurkat/
├── imagesTr/                    # 训练图像
├── instancesTr/                 # 训练标签
├── imagesTs/                    # 测试图像
├── instancesTs/                 # 测试标签（Ground Truth）
├── imagesTs_mitoNet_FT/         # MitoNet Fine-tuned预测 ⭐ NEW
│   ├── {filename}.tiff          # 最终预测结果
│   ├── {filename}_xy.tiff       # xy平面预测
│   ├── {filename}_xz.tiff       # xz平面预测
│   ├── {filename}_yz.tiff       # yz平面预测
│   └── evaluation/              # 评估结果
│       ├── evaluation_results.csv
│       └── evaluation_results_summary.txt
├── imagesTs_nnunet_pred/        # nnU-Net预测（对比）
├── imagesTs_microsam_pred/      # MicroSAM预测（对比）
└── ...
```

## 🔧 具体修改

### 代码修改

#### 1. `predict_3d()` 函数
- 修改最终预测文件命名：去掉 `_consensus` 后缀
- 修改平面预测文件命名：统一使用 `.tiff` 扩展名
- 自动去除 `_0000` 后缀

#### 2. `evaluate_predictions()` 函数
- 更新预测文件查找逻辑：支持 `.tiff` 和 `.tif` 格式
- 改进ground truth文件匹配逻辑
- 添加详细的评估结果输出

#### 3. `main()` 函数
- 添加 `--test_after_training` 参数
- 添加 `--test_axes` 参数（配置测试时使用的推理平面）
- 实现训练后自动测试流程
- 修改测试输出目录到 `{dataset_path}/imagesTs_mitoNet_FT`

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test_after_training` | flag | False | 训练后自动进行测试和评估 |
| `--test_axes` | list | [xy, xz, yz] | 测试时使用的推理平面 |

## 📝 使用示例

### 完整流程：训练 + 自动测试 + 评估

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

### 只使用xy平面进行快速测试

```bash
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    /path/to/dataset \
    checkpoints/my_model \
    --model_name MyModel \
    --iterations 1000 \
    --test_after_training \
    --test_axes xy
```

### 单独预测模式

```bash
conda run -n mitohard python src/training/mitoNet_finetune_from_nnunet.py \
    --mode predict \
    --trained_model checkpoints/my_model/MyModel.yaml \
    --input /path/to/test/images \
    --output /path/to/output \
    --use_gpu \
    --axes xy xz yz
```

## 📊 输出文件说明

### 文件命名示例

**输入文件**：`jrc_jurkat-1_recon-1_test1_0000.tiff`

**输出文件**：
- `jrc_jurkat-1_recon-1_test1.tiff` - 最终预测（去掉了_0000后缀）
- `jrc_jurkat-1_recon-1_test1_xy.tiff` - xy平面预测
- `jrc_jurkat-1_recon-1_test1_xz.tiff` - xz平面预测
- `jrc_jurkat-1_recon-1_test1_yz.tiff` - yz平面预测

### 评估报告

**位置**：`{dataset_path}/imagesTs_mitoNet_FT/evaluation/`

**文件**：
- `evaluation_results.csv` - 详细的逐文件评估结果
- `evaluation_results_summary.txt` - 汇总统计信息

**评估指标**：
- IoU (Intersection over Union)
- Dice Coefficient
- Precision
- Recall
- F1 Score
- 对象数量统计

## 🔍 与原有方法的对比

| 功能 | 原脚本 | 更新后 |
|------|--------|--------|
| 训练 | ✅ | ✅ |
| 预测 | ❌ | ✅ |
| 自动测试 | ❌ | ✅ |
| 自动评估 | ❌ | ✅ |
| 标准化输出 | ❌ | ✅ |
| 文件名处理 | - | ✅ 去_0000 |
| 与nnUNet对比 | 困难 | ✅ 目录并列 |

## 💡 优势

### 1. 一键完成完整流程
训练、预测、评估一条命令完成，无需手动操作

### 2. 标准化输出
预测结果保存在数据集目录下，与其他方法（nnU-Net、MicroSAM等）的预测结果并列，便于对比

### 3. 自动评估
训练完成即可看到模型在测试集上的性能，无需额外脚本

### 4. 文件名兼容性
输出文件名与nnU-Net保持一致（去掉_0000后缀），方便后续处理

### 5. 灵活配置
可以通过 `--test_axes` 参数选择只在单个平面预测（快速）或多平面共识（高质量）

## 🔧 技术细节

### 数据流程

```
1. 训练阶段
   nnUNet 3D TIFF → 2D Slices → MitoNet训练 → 保存模型

2. 测试阶段（如果启用 --test_after_training）
   imagesTs/*.tiff → 3D预测 → imagesTs_mitoNet_FT/*.tiff

3. 评估阶段
   imagesTs_mitoNet_FT/*.tiff + instancesTs/*.tiff → 评估指标 → evaluation/*.csv
```

### 错误处理

- ✅ 自动检查 `imagesTs` 和 `instancesTs` 是否存在
- ✅ 自动检查模型文件是否成功生成
- ✅ 预测和评估过程中的错误不会中断整体流程
- ✅ 详细的错误信息和警告提示

### 性能考虑

- GPU自动检测和使用
- 支持配置推理平面数量（平衡速度和质量）
- 支持下采样以减少内存使用

## 📚 相关文档

- **主文档**：`MITONET_FINETUNE_USAGE.md` - 完整使用指南
- **快速参考**：`README.md` - 项目主README
- **更新说明**：`UPDATE_SUMMARY.md` - 本文件

## 🎯 后续计划

- [ ] 添加更多评估指标（例如：边界准确度）
- [ ] 支持多GPU并行预测
- [ ] 添加预测进度条
- [ ] 生成可视化对比图
- [ ] 支持批量数据集的对比报告

## ✅ 测试建议

### 快速测试（5分钟）
```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    /path/to/dataset \
    ./test_model \
    --iterations 10 \
    --batch_size 4 \
    --test_after_training \
    --test_axes xy
```

### 标准测试（数小时）
```bash
python src/training/mitoNet_finetune_from_nnunet.py \
    /path/to/dataset \
    ./model \
    --iterations 1000 \
    --batch_size 8 \
    --test_after_training
```

## 🐛 故障排除

### 问题：找不到测试图像
**原因**：`imagesTs` 目录不存在或为空
**解决**：确保数据集包含 `imagesTs` 和 `instancesTs` 目录

### 问题：评估失败
**原因**：预测文件和ground truth文件名不匹配
**解决**：检查文件名格式，脚本会自动尝试多种匹配方式

### 问题：GPU内存不足
**解决**：
- 训练时：减小 `--batch_size`
- 预测时：使用 `--test_axes xy`（只用一个平面）

## 📞 支持

如有问题，请查看：
1. `MITONET_FINETUNE_USAGE.md` - 详细使用文档
2. 运行 `python src/training/mitoNet_finetune_from_nnunet.py --help` 查看所有参数

---

**更新完成！现在可以更方便地训练、测试和评估MitoNet模型了！** 🎉


