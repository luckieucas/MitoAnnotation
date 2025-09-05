# SLURM任务提交指南

这个目录包含了用于在SLURM集群上提交nnUNet训练任务的脚本。

## 📁 脚本文件

### 1. `submit_auto_training.sl` - 完整流程提交脚本
- **用途**: 提交完整的nnUNet训练流程
- **运行时间**: 48小时
- **资源**: 1个GPU, 64GB内存, 8个CPU核心
- **功能**: 自动运行从数据转换到结果评估的完整流程

### 2. `submit_step_by_step.sl` - 分步骤提交脚本
- **用途**: 提交单个步骤的任务
- **运行时间**: 24小时（可根据步骤调整）
- **资源**: 1个GPU, 64GB内存, 8个CPU核心
- **功能**: 支持运行特定的训练步骤

### 3. `submit_jobs.sh` - 便捷提交脚本
- **用途**: 简化SLURM任务提交过程
- **功能**: 根据任务类型自动选择合适的SLURM脚本

## 🚀 使用方法

### 方法1: 使用便捷脚本（推荐）

```bash
# 提交完整训练流程
./submit_jobs.sh full

# 只转换数据
./submit_jobs.sh convert

# 只训练模型
./submit_jobs.sh train

# 只运行预测
./submit_jobs.sh predict
```

### 方法2: 直接使用SLURM脚本

```bash
# 提交完整流程
sbatch scripts/submit_auto_training.sl

# 提交特定步骤
sbatch scripts/submit_step_by_step.sl convert
sbatch scripts/submit_step_by_step.sl train
sbatch scripts/submit_step_by_step.sl predict
```

## 📋 可用的任务类型

| 任务类型 | 描述 | 预估时间 | 资源需求 |
|----------|------|----------|----------|
| `full` | 完整训练流程 | 48小时 | 高 |
| `convert` | 数据转换 (h5→tiff) | 4小时 | 低 |
| `dataset` | 创建nnUNet数据集 | 2小时 | 低 |
| `boundary` | 生成边界mask | 2小时 | 低 |
| `plan` | 数据预处理 | 8小时 | 中 |
| `train` | 模型训练 | 24小时 | 高 |
| `predict` | 模型预测 | 4小时 | 中 |
| `postprocess` | 后处理 | 2小时 | 低 |
| `evaluate` | 结果评估 | 1小时 | 低 |

## 🔧 配置说明

### 环境配置
所有脚本都配置为：
- **分区**: `weilab`
- **节点**: 1个
- **GPU**: 1个
- **内存**: 64GB
- **CPU**: 8核心
- **Conda环境**: `sam2_mito`

### 路径配置
- **数据文件夹**: `/projects/weilab/dataset/MitoLE/betaSeg`
- **输出目录**: `/projects/weilab/liupeng/MitoAnnotation/nnunet_data`
- **工作目录**: `/projects/weilab/liupeng/MitoAnnotation`

## 📊 监控任务

### 查看任务状态
```bash
# 查看所有任务
squeue -u $USER

# 查看特定任务详情
scontrol show job <job_id>
```

### 查看日志
```bash
# 查看输出日志
tail -f logs/<job_id>_auto_nnunet.out

# 查看错误日志
tail -f logs/<job_id>_auto_nnunet.err
```

### 取消任务
```bash
scancel <job_id>
```

## ⚠️ 注意事项

### 1. 资源限制
- 确保你的账户有足够的GPU配额
- 检查分区`weilab`的可用性
- 根据数据大小调整内存需求

### 2. 数据准备
- 确保数据文件夹存在且包含正确的文件
- 检查文件命名规范（`*_im.h5`, `*_mito.h5`）
- 验证数据格式和完整性

### 3. 环境依赖
- 确保`sam2_mito` conda环境已安装
- 检查nnUNet和相关依赖是否正确安装
- 验证CUDA环境配置

## 🔍 故障排除

### 常见问题

1. **任务被拒绝**
   ```bash
   # 检查分区状态
   sinfo -p weilab
   
   # 检查资源可用性
   squeue -p weilab
   ```

2. **GPU内存不足**
   - 减少batch size
   - 使用较小的patch size
   - 检查GPU使用情况

3. **任务超时**
   - 根据数据大小调整时间限制
   - 分步骤运行而不是完整流程
   - 监控任务进度

4. **环境问题**
   ```bash
   # 检查conda环境
   conda info --envs
   
   # 重新激活环境
   conda activate sam2_mito
   ```

### 调试模式
如果遇到问题，可以：
1. 先运行简单的步骤（如`convert`）
2. 检查日志文件中的错误信息
3. 验证环境变量设置
4. 测试单个组件的功能

## 📈 性能优化

### 1. 资源利用
- 根据数据大小调整内存分配
- 使用合适的CPU核心数
- 监控GPU利用率

### 2. 时间管理
- 分步骤运行便于调试
- 根据步骤复杂度调整时间限制
- 并行运行独立的步骤

### 3. 存储优化
- 使用SSD存储预处理数据
- 定期清理临时文件
- 监控磁盘空间使用

## 🔄 工作流程建议

### 首次运行
1. 先运行`convert`验证数据转换
2. 运行`dataset`检查数据集创建
3. 运行`plan`进行数据预处理
4. 运行`train`开始模型训练

### 调试流程
1. 使用较小的数据集测试
2. 分步骤运行并检查结果
3. 监控资源使用情况
4. 及时调整参数配置

### 生产运行
1. 使用完整流程脚本
2. 设置适当的监控和通知
3. 定期检查任务状态
4. 备份重要的中间结果


