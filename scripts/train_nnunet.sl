#!/bin/bash
#SBATCH --job-name=train_nnUNet        # Job name
#SBATCH --time=24:00:00                # Maximum runtime
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=6              # Number of CPU cores per task
#SBATCH --gres=gpu:1                   # Number of GPUs per task
#SBATCH --mem=32gb                     # Memory allocation
#SBATCH --partition=weilab             # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications (on job start, end, or fail)
#SBATCH --mail-user=liupen@bc.edu      # Email address for notifications
#SBATCH --output=logs/%j_nnunet.out  

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate sam2_mito

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
nnUNetv2_train $TASK_ID 3d_fullres all

# Print end time
echo "Training completed at $(date)"
