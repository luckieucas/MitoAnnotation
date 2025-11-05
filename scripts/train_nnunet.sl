#!/bin/bash
#SBATCH --job-name=train_nnUNet        # Job name
#SBATCH --time=12:00:00                # Maximum runtime
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=6              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs per task
#SBATCH --mem=64gb                     # Memory allocation
#SBATCH --partition=short             # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications (on job start, end, or fail)
#SBATCH --mail-user=liupen@bc.edu      # Email address for notifications
#SBATCH --output=logs/%j_nnunet.out  

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate mitoem2

MY_TMPDIR="/projects/weilab/liupeng/nnunet_tmpdir/${SLURM_JOB_ID}"
mkdir -p $MY_TMPDIR

# 2. 导出 TMPDIR 环境变量，让 g++ 和 torch.compile 使用这个新路径
export TMPDIR=$MY_TMPDIR

# Check if task ID is provided
if [ -z "$1" ]; then
  echo "Error: No task ID provided. Usage: sbatch train.sh <task_id>"
  exit 1
fi

TASK_ID=$1

# Print debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Using nnUNet task ID: $TASK_ID"

# Run the nnUNet training command with specified task ID
#nnUNetv2_plan_and_preprocess -d $TASK_ID --verify_dataset_integrity -c 3d_fullreså

nnUNetv2_train $TASK_ID 3d_fullres 0

# Print end time
echo "Training completed at $(date)"
