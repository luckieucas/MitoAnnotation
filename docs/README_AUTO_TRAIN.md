# 自动训练nnunet完整流程

这个项目提供了一个完整的自动化流程来训练nnunet模型，专门用于线粒体分割任务。

## 功能概述

自动训练脚本 `auto_train_nnunet.py` 包含以下完整流程：

1. **数据转换**: 将h5格式的图像和mask文件转换为tiff格式并直接保存到nnunet数据集
2. **数据集创建**: 创建符合nnunet要求的数据集结构
3. **边界生成**: 使用 `generate_contour.py` 生成带boundary的mask
4. **数据预处理**: 运行nnunet plan and process
5. **模型训练**: 使用nnunet训练分割模型
6. **模型预测**: 对新数据进行预测
7. **后处理**: 去掉boundary，运行 `bc_watershed.py` 进行实例分割
8. **结果评估**: 使用 `evaluate_res.py` 评估分割结果

## 环境要求

### 必需软件
- Python 3.7+
- nnunet (安装并配置)
- CUDA (如果使用GPU训练)

### Python依赖
```bash
pip install -r requirements.txt
```

### nnunet安装
```bash
pip install nnunet
```

## 使用方法

### 方法1: 直接调用函数（推荐）

#### 在Python脚本中使用
```python
from auto_train_nnunet import AutoTrainNnunet

# 创建训练器实例
trainer = AutoTrainNnunet(
    data_folder="/projects/weilab/dataset/MitoLE/betaSeg",
    output_base_dir="/projects/weilab/liupeng/MitoAnnotation/nnunet_data",
    dataset_id="Dataset001_MitoLE"
)

# 运行完整流程
trainer.run_full_pipeline(
    fold=0,
    trainer="nnUNetTrainer",
    max_epochs=1000
)
```

#### 分步骤运行
```python
# 步骤1: 转换数据并创建数据集
trainer.convert_h5_to_tiff_direct()

# 步骤2: 生成boundary masks
trainer.generate_boundary_masks_direct()

# 步骤3: 运行nnunet plan and process
trainer.nnunet_plan_and_process()

# 步骤4: 训练模型
trainer.nnunet_train(fold=0, trainer="nnUNetTrainer", max_epochs=1000)

# 步骤5: 运行预测
trainer.nnunet_predict(fold=0, trainer="nnUNetTrainer")

# 步骤6: 后处理
trainer.postprocess_predictions()

# 步骤7: 评估结果
trainer.evaluate_results()
```

### 方法2: 使用示例脚本

#### 简单示例
```bash
python simple_training_example.py
```

#### 交互式示例
```bash
python run_training_direct.py
```

### 方法3: 使用Jupyter Notebook
```bash
jupyter notebook notebook_example.ipynb
```

### 方法4: 命令行方式（传统方式）

#### 设置环境变量
```bash
source setup_nnunet_env.sh
```

#### 运行完整流程
```bash
python auto_train_nnunet.py \
    --data_folder "/projects/weilab/dataset/MitoLE/betaSeg" \
    --output_dir "/projects/weilab/liupeng/MitoAnnotation/nnunet_data" \
    --dataset_id "Dataset001_MitoLE" \
    --fold 0 \
    --trainer "nnUNetTrainer" \
    --max_epochs 1000
```

### 方法5: 使用SLURM脚本

#### 完整流程
```bash
sbatch scripts/submit_auto_training.sl
```

#### 分步骤运行
```bash
./scripts/submit_jobs.sh convert    # 只转换数据
./scripts/submit_jobs.sh train      # 只训练模型
./scripts/submit_jobs.sh predict    # 只运行预测
```

## 主要改进

### 1. 直接函数调用
- 不再需要通过命令行参数
- 可以直接在Python代码中调用
- 支持Jupyter notebook使用

### 2. 优化的数据流程
- 不保存中间的`tiff_converted`文件
- 直接将转换后的数据保存到nnunet数据集目录
- 图像保存到`imagesTr`，标签保存到`labelsTr`

### 3. 智能boundary生成
- 优先使用Python函数直接生成
- 如果导入失败，自动回退到命令行方式
- 直接更新标签文件，无需复制

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `data_folder` | str | 是 | - | 包含h5文件的输入文件夹路径 |
| `output_base_dir` | str | 是 | - | 输出基础目录 |
| `dataset_id` | str | 否 | "Dataset001_MitoLE" | nnunet数据集ID |

## 输出目录结构

运行完成后，输出目录将包含：

```
nnunet_data/
├── nnunet_dataset/          # nnunet数据集
│   └── Dataset001_MitoLE/
│       ├── imagesTr/        # 训练图像（直接从h5转换）
│       ├── labelsTr/        # 训练标签（带boundary）
│       ├── imagesTs/        # 测试图像
│       └── dataset.json     # 数据集配置文件
├── boundary_masks/          # 生成的boundary masks（临时）
├── preprocessed/            # nnunet预处理数据
├── results/                 # 训练结果和模型
├── predictions/             # 模型预测结果
└── final_results/           # 后处理后的最终结果
```

## 训练配置

### 默认训练参数
- **模型**: 3D Full Resolution
- **训练器**: nnUNetTrainer
- **数据增强**: 默认nnunet设置
- **优化器**: SGD with momentum
- **损失函数**: Dice + Cross Entropy

### 自定义训练参数
你可以修改训练参数：

```python
# 使用不同的trainer
trainer.nnunet_train(
    fold=0,
    trainer="nnUNetTrainerNoMirroring",  # 不使用镜像增强
    max_epochs=500
)
```

## 监控和日志

脚本会自动生成详细的日志文件：
- `auto_train_nnunet.log`: 主要日志文件
- 控制台输出: 实时进度信息

### 日志级别
- INFO: 主要步骤信息
- DEBUG: 详细调试信息
- ERROR: 错误信息

## 故障排除

### 常见问题

1. **nnunet命令未找到**
   ```bash
   pip install nnunet
   # 或者检查PATH环境变量
   ```

2. **CUDA内存不足**
   - 减少batch size
   - 使用较小的patch size
   - 检查GPU内存使用情况

3. **数据格式错误**
   - 确保h5文件包含正确的数据集
   - 检查图像和mask的维度匹配
   - 验证文件命名规范

4. **connectomics模块导入失败**
   - 脚本会自动回退到命令行方式
   - 确保`generate_contour.py`脚本可用

### 调试模式
启用详细日志：
```python
logging.getLogger().setLevel(logging.DEBUG)
```

## 性能优化建议

1. **数据预处理**
   - 使用SSD存储预处理数据
   - 确保足够的磁盘空间

2. **训练优化**
   - 使用多GPU训练（如果可用）
   - 调整batch size和patch size
   - 使用混合精度训练

3. **内存管理**
   - 监控GPU内存使用
   - 适当调整数据加载器的工作进程数

## 扩展和定制

### 添加新的数据增强
修改 `generate_boundary_masks_direct` 方法中的参数：
```python
# 调整boundary宽度
contour = seg_to_instance_bd(binary, tsz_h=5)  # 增加boundary宽度
```

### 自定义评估指标
修改 `evaluate_results` 方法，添加新的评估指标。

### 集成其他模型
可以扩展脚本支持其他分割模型，如nnunet2、swin-unetr等。

## 技术支持

如果遇到问题，请检查：
1. 日志文件中的错误信息
2. nnunet的官方文档
3. 环境配置是否正确
4. 数据格式是否符合要求

## 更新日志

- v2.0: 支持直接函数调用，优化数据流程
  - 移除中间文件保存
  - 直接保存到nnunet数据集目录
  - 支持Jupyter notebook使用
  - 智能boundary生成
- v1.0: 初始版本，支持完整的nnunet训练流程
  - 支持h5到tiff的自动转换
  - 集成boundary generation和bc_watershed后处理
  - 自动化评估流程


