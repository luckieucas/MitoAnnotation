#!/bin/bash
# 示例脚本：训练和评估 Dataset004_MitoHardCardiac

# 设置路径
DATASET=/projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset004_MitoHardCardiac
MODEL_DIR=/projects/weilab/liupeng/MitoAnnotation/trained_models/cardiac_mitonet
DATA_2D=/projects/weilab/liupeng/dataset/mito/cardiac_2d_slices

# 创建目录
mkdir -p $MODEL_DIR
mkdir -p $DATA_2D

# 进入脚本所在目录的父目录
cd "$(dirname "$0")/.."

echo "Starting MitoNet fine-tuning for Cardiac dataset..."
echo "=================================================="

# 1. 训练模型（包含数据转换）
echo ""
echo "Step 1: Training model (with data conversion)..."
python src/training/mitoNet_finetune_from_nnunet.py \
    $DATASET \
    $MODEL_DIR \
    --output_data_path $DATA_2D \
    --model_name MitoNet_Cardiac \
    --iterations 1000 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --finetune_layer all

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo ""
echo "Step 2: Evaluating model..."
# 2. 评估模型
python src/evaluation/evaluate_mitonet.py \
    $MODEL_DIR/MitoNet_Cardiac.yaml \
    $DATA_2D/val/raw \
    $DATA_2D/val/labels \
    --output_dir $MODEL_DIR/evaluation_results

if [ $? -ne 0 ]; then
    echo "Evaluation failed!"
    exit 1
fi

echo ""
echo "=================================================="
echo "Training and evaluation completed!"
echo "Model saved to: $MODEL_DIR"
echo "Evaluation results: $MODEL_DIR/evaluation_results"
echo "=================================================="



