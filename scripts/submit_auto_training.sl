#!/bin/bash
#SBATCH --job-name=auto_nnUNet        # Job name
#SBATCH --time=48:00:00               # Maximum runtime (48 hours for full pipeline)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs per task
#SBATCH --mem=64gb                    # Memory allocation (increased for full pipeline)
#SBATCH --partition=weilab            # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH --mail-user=liupen@bc.edu     # Email address for notifications
#SBATCH --output=logs/%j_auto_nnunet.out
#SBATCH --error=logs/%j_auto_nnunet.err

# Activate the Conda environment
source ~/miniconda3/bin/activate
conda activate sam2_mito

# Create logs directory if it doesn't exist
mkdir -p logs

# Set working directory
cd /projects/weilab/liupeng/MitoAnnotation

# Print debug information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Check if GPU is available
nvidia-smi
echo "GPU memory before training:"
nvidia-smi --query-gpu=memory.used --format=csv

# Set environment variables for nnUNet
export nnUNet_raw_data_base="/projects/weilab/liupeng/MitoAnnotation/nnunet_data"
export nnUNet_preprocessed="/projects/weilab/liupeng/MitoAnnotation/nnunet_data/preprocessed"
export RESULTS_FOLDER="/projects/weilab/liupeng/MitoAnnotation/nnunet_data/results"

# Create necessary directories
mkdir -p "$nnUNet_raw_data_base"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$RESULTS_FOLDER"

# Data paths
DATA_FOLDER="/projects/weilab/dataset/MitoLE/betaSeg"
OUTPUT_DIR="/projects/weilab/liupeng/MitoAnnotation/nnunet_data"

echo "Data folder: $DATA_FOLDER"
echo "Output directory: $OUTPUT_DIR"

# Check if data folder exists
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: Data folder does not exist: $DATA_FOLDER"
    exit 1
fi

# Check data files
echo "Checking data files..."
H5_FILES=$(find "$DATA_FOLDER" -name "*.h5" | wc -l)
if [ "$H5_FILES" -eq 0 ]; then
    echo "Error: No .h5 files found in data folder"
    exit 1
fi
echo "Found $H5_FILES .h5 files"

# Start the automatic training pipeline
echo "Starting automatic training pipeline..."
echo "This will run the complete pipeline: convert -> create_dataset -> generate_boundary -> plan_process -> train -> predict -> postprocess -> evaluate"
echo ""

python auto_train_nnunet.py \
    --data_folder "$DATA_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_id "Dataset001_MitoLE" \
    --fold 0 \
    --trainer "nnUNetTrainer" \
    --max_epochs 1000

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training pipeline completed successfully! ==="
    echo "Results saved in: $OUTPUT_DIR"
    echo "Log file: auto_train_nnunet.log"
    
    # Show final GPU memory usage
    echo "GPU memory after training:"
    nvidia-smi --query-gpu=memory.used --format=csv
    
    # List output files
    echo "Output files:"
    ls -la "$OUTPUT_DIR"
    
else
    echo ""
    echo "=== Training pipeline failed! ==="
    echo "Check the log file: auto_train_nnunet.log"
    exit 1
fi

echo ""
echo "Job completed at $(date)"


