#!/bin/bash
#SBATCH --job-name=complexity_eval   # Job name
#SBATCH --time=1:00:00           # Maximum runtime
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --gres=gpu:1              # 申请1个GPU
#SBATCH --gpus-per-task=1         # 每个任务分配1个GPU
#SBATCH --cpus-per-task=32         # Number of CPU cores per task
#SBATCH --mem=64gb                # Memory allocation
#SBATCH --partition=short# Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications (on job start, end, or fail)
#SBATCH --mail-user=liupen@bc.edu  # Email address for notifications
#SBATCH --output=logs/%j_complexity_eval.out     # Output log file (%j represents the Job ID)

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate sam2_mito

# Print debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

# Run the nnUNet training command
#python -u evaluate_dataset_difficult.py  /projects/weilab/dataset/MitoLE --output-dir /projects/weilab/liupeng/dataset/mito/complexity/ --datasets-hard  cellmap24_cardiac betaSeg han24 wei20 mitoNet_hard cellmap24_jurkat Kedar_4yv2h Kedar_f536  --datasets-easy mitoNet_easy urocell casser20 --compute-contact-count

python -u evaluate_dataset_difficult.py  /projects/weilab/dataset/MitoLE --output-dir /projects/weilab/liupeng/dataset/mito/complexity/ --datasets-hard  cellmap24_kidney cellmap24_liver --compute-contact-count
# Print end time
echo "Training completed at $(date)"
