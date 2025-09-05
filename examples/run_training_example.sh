#!/bin/bash
# 自动训练nnunet的示例脚本

# 设置环境变量
source setup_nnunet_env.sh

# 数据文件夹路径
DATA_FOLDER="/projects/weilab/dataset/MitoLE/betaSeg"

# 输出目录
OUTPUT_DIR="/projects/weilab/liupeng/MitoAnnotation/nnunet_data"

# 运行完整的训练流程
python auto_train_nnunet.py \
    --data_folder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_id "Dataset001_MitoLE" \
    --fold 0 \
    --trainer "nnUNetTrainer" \
    --max_epochs 1000

echo "训练流程完成！"


