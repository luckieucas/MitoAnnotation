#!/bin/bash
# 批量处理多个nnUNet数据集

# 定义数据集列表
DATASETS=(
    "Dataset004_MitoHardCardiac"
    "Dataset005_MitoHardKidney"
    "Dataset006_MitoHardLiver"
)

BASE_PATH="/projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw"
OUTPUT_BASE="/projects/weilab/liupeng/MitoAnnotation/trained_models"
DATA_BASE="/projects/weilab/liupeng/dataset/mito"

# 进入脚本所在目录的父目录
cd "$(dirname "$0")/.."

# 训练参数
ITERATIONS=1000
BATCH_SIZE=8
LEARNING_RATE=0.001

echo "========================================"
echo "Batch Processing MitoNet Fine-tuning"
echo "========================================"
echo "Datasets to process: ${#DATASETS[@]}"
echo "Iterations: $ITERATIONS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "========================================"

# 循环处理每个数据集
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing $dataset..."
    echo "========================================"
    
    DATASET_PATH="$BASE_PATH/$dataset"
    MODEL_DIR="$OUTPUT_BASE/${dataset}_mitonet"
    DATA_2D="$DATA_BASE/${dataset}_2d_slices"
    
    # 检查数据集是否存在
    if [ ! -d "$DATASET_PATH" ]; then
        echo "Warning: Dataset not found at $DATASET_PATH, skipping..."
        continue
    fi
    
    # 创建输出目录
    mkdir -p $MODEL_DIR
    mkdir -p $DATA_2D
    
    echo ""
    echo "Step 1: Training $dataset..."
    echo "----------------------------------------"
    # 训练
    python src/training/mitoNet_finetune_from_nnunet.py \
        $DATASET_PATH \
        $MODEL_DIR \
        --output_data_path $DATA_2D \
        --model_name MitoNet_${dataset} \
        --iterations $ITERATIONS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --finetune_layer all
    
    if [ $? -ne 0 ]; then
        echo "Training failed for $dataset!"
        echo "Continuing to next dataset..."
        continue
    fi
    
    echo ""
    echo "Step 2: Evaluating $dataset..."
    echo "----------------------------------------"
    # 评估
    python src/evaluation/evaluate_mitonet.py \
        $MODEL_DIR/MitoNet_${dataset}.yaml \
        $DATA_2D/val/raw \
        $DATA_2D/val/labels \
        --output_dir $MODEL_DIR/evaluation_results
    
    if [ $? -ne 0 ]; then
        echo "Evaluation failed for $dataset!"
        echo "Continuing to next dataset..."
        continue
    fi
    
    echo ""
    echo "Completed $dataset successfully!"
    echo "Model: $MODEL_DIR/MitoNet_${dataset}.yaml"
    echo "Results: $MODEL_DIR/evaluation_results/evaluation_results_summary.txt"
    echo "========================================"
done

echo ""
echo "========================================"
echo "All datasets processed!"
echo "========================================"

# 生成汇总报告
echo ""
echo "Generating summary report..."
SUMMARY_FILE="$OUTPUT_BASE/all_results_summary.txt"
echo "=======================================" > $SUMMARY_FILE
echo "MitoNet Fine-tuning Summary Report" >> $SUMMARY_FILE
echo "=======================================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

for dataset in "${DATASETS[@]}"; do
    MODEL_DIR="$OUTPUT_BASE/${dataset}_mitonet"
    RESULTS_FILE="$MODEL_DIR/evaluation_results/evaluation_results_summary.txt"
    
    echo "----------------------------------------" >> $SUMMARY_FILE
    echo "$dataset" >> $SUMMARY_FILE
    echo "----------------------------------------" >> $SUMMARY_FILE
    
    if [ -f "$RESULTS_FILE" ]; then
        cat "$RESULTS_FILE" >> $SUMMARY_FILE
    else
        echo "No results found" >> $SUMMARY_FILE
    fi
    
    echo "" >> $SUMMARY_FILE
done

echo "Summary report saved to: $SUMMARY_FILE"
cat $SUMMARY_FILE



