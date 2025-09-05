#!/bin/bash
# 提交SLURM任务的便捷脚本

echo "=== nnUNet SLURM任务提交脚本 ==="
echo ""

# 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <任务类型>"
    echo ""
    echo "可用的任务类型:"
    echo "  full        - 完整训练流程 (48小时)"
    echo "  convert     - 只转换数据 (4小时)"
    echo "  dataset     - 创建数据集 (2小时)"
    echo "  boundary    - 生成边界 (2小时)"
    echo "  plan        - 数据预处理 (8小时)"
    echo "  train       - 训练模型 (24小时)"
    echo "  predict     - 模型预测 (4小时)"
    echo "  postprocess - 后处理 (2小时)"
    echo "  evaluate    - 评估结果 (1小时)"
    echo ""
    echo "示例:"
    echo "  $0 full        # 提交完整流程"
    echo "  $0 convert     # 只转换数据"
    echo "  $0 train       # 只训练模型"
    exit 1
fi

TASK_TYPE=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case $TASK_TYPE in
    "full")
        echo "提交完整训练流程任务..."
        sbatch "$SCRIPT_DIR/submit_auto_training.sl"
        ;;
    "convert"|"dataset"|"boundary"|"plan"|"train"|"predict"|"postprocess"|"evaluate")
        echo "提交步骤任务: $TASK_TYPE"
        sbatch "$SCRIPT_DIR/submit_step_by_step.sl" "$TASK_TYPE"
        ;;
    *)
        echo "错误: 未知的任务类型 '$TASK_TYPE'"
        echo "请使用 'full', 'convert', 'dataset', 'boundary', 'plan', 'train', 'predict', 'postprocess', 或 'evaluate'"
        exit 1
        ;;
esac

echo ""
echo "任务已提交！"
echo "使用 'squeue -u $USER' 查看任务状态"
echo "使用 'scontrol show job <job_id>' 查看任务详情"


