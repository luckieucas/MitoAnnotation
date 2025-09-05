#!/bin/bash
#SBATCH --job-name=nnUNet_step        # Job name
#SBATCH --time=24:00:00               # Maximum runtime
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs per task
#SBATCH --mem=64gb                    # Memory allocation
#SBATCH --partition=weilab            # Partition to submit the job
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH --mail-user=liupen@bc.edu     # Email address for notifications
#SBATCH --output=logs/%j_step_%a.out
#SBATCH --error=logs/%j_step_%a.err

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

# Check if step argument is provided
if [ -z "$1" ]; then
    echo "Error: No step specified. Usage: sbatch submit_step_by_step.sl <step_name>"
    echo "Available steps:"
    echo "  convert      - Convert h5 to tiff"
    echo "  dataset      - Create nnunet dataset"
    echo "  boundary     - Generate boundary masks"
    echo "  plan         - Run nnunet plan and process"
    echo "  train        - Train nnunet model"
    echo "  predict      - Run prediction"
    echo "  postprocess  - Post-process results"
    echo "  evaluate     - Evaluate results"
    echo "  full         - Run full pipeline"
    exit 1
fi

STEP=$1

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

echo "Step: $STEP"
echo "Data folder: $DATA_FOLDER"
echo "Output directory: $OUTPUT_DIR"

# Check if data folder exists
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: Data folder does not exist: $DATA_FOLDER"
    exit 1
fi

# Function to run specific step
run_step() {
    local step=$1
    case $step in
        "convert")
            echo "Running step: Convert h5 to tiff..."
            python auto_train_nnunet.py \
                --data_folder "$DATA_FOLDER" \
                --output_dir "$OUTPUT_DIR" \
                --dataset_id "Dataset001_MitoLE" \
                --skip_training \
                --skip_prediction
            ;;
        "dataset")
            echo "Running step: Create nnunet dataset..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.convert_h5_to_tiff()
trainer.create_nnunet_dataset()
print('Dataset creation completed')
"
            ;;
        "boundary")
            echo "Running step: Generate boundary masks..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.generate_boundary_masks()
print('Boundary generation completed')
"
            ;;
        "plan")
            echo "Running step: nnunet plan and process..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.nnunet_plan_and_process()
print('Plan and process completed')
"
            ;;
        "train")
            echo "Running step: Train nnunet model..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.nnunet_train(fold=0, trainer='nnUNetTrainer', max_epochs=1000)
print('Training completed')
"
            ;;
        "predict")
            echo "Running step: Run prediction..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.nnunet_predict(fold=0, trainer='nnUNetTrainer')
print('Prediction completed')
"
            ;;
        "postprocess")
            echo "Running step: Post-process results..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.postprocess_predictions()
print('Post-processing completed')
"
            ;;
        "evaluate")
            echo "Running step: Evaluate results..."
            python -c "
from auto_train_nnunet import AutoTrainNnunet
trainer = AutoTrainNnunet('$DATA_FOLDER', '$OUTPUT_DIR', 'Dataset001_MitoLE')
trainer.evaluate_results()
print('Evaluation completed')
"
            ;;
        "full")
            echo "Running full pipeline..."
            python auto_train_nnunet.py \
                --data_folder "$DATA_FOLDER" \
                --output_dir "$OUTPUT_DIR" \
                --dataset_id "Dataset001_MitoLE" \
                --fold 0 \
                --trainer "nnUNetTrainer" \
                --max_epochs 1000
            ;;
        *)
            echo "Error: Unknown step '$step'"
            exit 1
            ;;
    esac
}

# Run the specified step
echo "Starting step: $STEP"
run_step "$STEP"

# Check if step was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Step '$STEP' completed successfully! ==="
    
    # Show output files for certain steps
    if [ "$STEP" = "convert" ] || [ "$STEP" = "dataset" ] || [ "$STEP" = "boundary" ]; then
        echo "Output files:"
        ls -la "$OUTPUT_DIR"
    fi
    
else
    echo ""
    echo "=== Step '$STEP' failed! ==="
    echo "Check the log file: auto_train_nnunet.log"
    exit 1
fi

echo ""
echo "Step '$STEP' completed at $(date)"


