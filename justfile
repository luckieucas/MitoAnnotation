# Default target: list available recipes
default:
    just --list

# -------------------------------------------------------------------
# Example training recipe (kept as in your original file)
# -------------------------------------------------------------------
train_mitonet_FT:
    uv run BANIS.py --seed 0 --batch_size 8 --n_steps 50000 --data_setting base --base_data_path /projects/weilab/dataset/nisb --save_path ./outputs --devices=1

# -------------------------------------------------------------------
# Example evaluation recipe (kept as in your original file)
# -------------------------------------------------------------------
evaluate:
    uv run src/evaluation/evaluate_res.py --pred_file /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/imagesTs_mitoNet_FT/jrc_jurkat-1_recon-1_test1_xy.tiff --gt_file /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/instancesTs/jrc_jurkat-1_recon-1_test1.tiff

# -------------------------------------------------------------------
# MitoNet baseline (local run)
# Now uses YAML config files for cleaner parameter management
#
# Usage examples:
#   just mitonet_baseline                          -> use default config
#   just mitonet_baseline 2                        -> override dataset to 2
#   just mitonet_baseline configs/custom.yaml      -> use custom config file
#   just mitonet_baseline 2 configs/custom.yaml    -> use custom config, override dataset
# -------------------------------------------------------------------
mitonet_baseline dataset_or_config="1" config_file="configs/mitonet_baseline_default.yaml":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f "{{dataset_or_config}}" ]]; then
        echo "Running MitoNet baseline with config: {{dataset_or_config}}"
        python src/inference/MitoNet_baseline.py --config "{{dataset_or_config}}"
    else
        echo "Running MitoNet baseline with dataset={{dataset_or_config}}, config={{config_file}}"
        python src/inference/MitoNet_baseline.py --config "{{config_file}}" -d "{{dataset_or_config}}"
    fi

# -------------------------------------------------------------------
# Submit MitoNet baseline to Slurm
#
# Usage examples:
#   just launch_mitonet_baseline                    # d=1, default time/partition
#   just launch_mitonet_baseline 2                  # positional arg: d=2
#   just launch_mitonet_baseline 2 "12:00:00" "short"
#   just launch_mitonet_baseline configs/custom.yaml "12:00:00"  # use custom config
# -------------------------------------------------------------------
default_time := "6:00:00"
default_partition := "weilab"

launch_mitonet_baseline dataset_or_config="1" time=default_time partition=default_partition:
    @echo "Submitting Slurm job for mitonet_baseline (dataset/config={{dataset_or_config}}) to partition {{partition}}, time {{time}}"
    @mkdir -p logs
    python submit_slurm.py -t {{time}} -p {{partition}} --job_name "mitonet_Baseline_{{dataset_or_config}}" --command "just mitonet_baseline {{dataset_or_config}}"

# -------------------------------------------------------------------
# MicroSAM baseline (local run)
# Now uses YAML config files for cleaner parameter management
#
# Usage examples:
#   just microsam_baseline                          -> use default config
#   just microsam_baseline 2                        -> override dataset to 2
#   just microsam_baseline configs/custom.yaml      -> use custom config file
# -------------------------------------------------------------------
microsam_baseline dataset_or_config="1" config_file="configs/microsam_baseline_default.yaml":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f "{{dataset_or_config}}" ]]; then
        echo "Running MicroSAM baseline with config: {{dataset_or_config}}"
        python src/inference/micro_sam_baseline.py --config "{{dataset_or_config}}"
    else
        echo "Running MicroSAM baseline with dataset={{dataset_or_config}}, config={{config_file}}"
        python src/inference/micro_sam_baseline.py --config "{{config_file}}" -d "{{dataset_or_config}}"
    fi

# -------------------------------------------------------------------
# Submit MicroSAM baseline to Slurm
#
# Usage examples:
#   just launch_microsam_baseline                    # d=1, default time/partition
#   just launch_microsam_baseline 2                  # positional arg: d=2
#   just launch_microsam_baseline 2 "12:00:00" "short"
# -------------------------------------------------------------------
launch_microsam_baseline dataset_or_config="1" time=default_time partition=default_partition:
    @echo "Submitting Slurm job for microsam_baseline (dataset/config={{dataset_or_config}}) to partition {{partition}}, time {{time}}"
    @mkdir -p logs
    python submit_slurm.py -t {{time}} -p {{partition}} --env_name sam --job_name "microsam_Baseline_{{dataset_or_config}}" --command "just microsam_baseline {{dataset_or_config}}"

# -------------------------------------------------------------------
# MitoNet Fine-tuning (local run)
# Now uses YAML config files for cleaner parameter management
#
# Usage examples:
#   just mitonet_finetune                          -> use default config
#   just mitonet_finetune 2                        -> override dataset to 2
#   just mitonet_finetune configs/custom.yaml      -> use custom config file
# -------------------------------------------------------------------
mitonet_finetune dataset_or_config="1" config_file="configs/mitonet_finetune_default.yaml":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f "{{dataset_or_config}}" ]]; then
        echo "Running MitoNet fine-tuning with config: {{dataset_or_config}}"
        python src/training/mitoNet_finetune.py --config "{{dataset_or_config}}"
    else
        echo "Running MitoNet fine-tuning with dataset={{dataset_or_config}}, config={{config_file}}"
        python src/training/mitoNet_finetune.py --config "{{config_file}}" -d "{{dataset_or_config}}"
    fi

# -------------------------------------------------------------------
# Submit MitoNet fine-tuning to Slurm
#
# Usage examples:
#   just launch_mitonet_finetune                    # d=1, default time/partition
#   just launch_mitonet_finetune 2                  # positional arg: d=2
#   just launch_mitonet_finetune 2 "24:00:00" "long"
# -------------------------------------------------------------------
launch_mitonet_finetune dataset_or_config="1" time=default_time partition=default_partition:
    @echo "Submitting Slurm job for mitonet_finetune (dataset/config={{dataset_or_config}}) to partition {{partition}}, time {{time}}"
    @mkdir -p logs
    python submit_slurm.py -t {{time}} -p {{partition}} --job_name "mitonet_Finetune_{{dataset_or_config}}" --command "just mitonet_finetune {{dataset_or_config}}"

# -------------------------------------------------------------------
# nnUNet Training (local run)
# Train nnUNet on a dataset from h5 files
#
# Usage examples:
#   just nnunet_train /path/to/data /path/to/output Dataset001_MitoLE
#   just nnunet_train /path/to/data /path/to/output Dataset002_Test 1
# -------------------------------------------------------------------
nnunet_train data_folder output_dir dataset_id fold="all":
    @echo "Running nnUNet training with dataset={{dataset_id}}, fold={{fold}}"
    python src/training/auto_train_nnunet.py --data_folder {{data_folder}} --output_dir {{output_dir}} --dataset_id {{dataset_id}} --fold {{fold}} 

# -------------------------------------------------------------------
# Submit nnUNet training to Slurm using submit_slurm.py wrapper.
#
# This recipe passes a quoted command string to the wrapper.
#
# Usage examples:
#   just launch_nnunet_train /path/to/data /path/to/output Dataset001_MitoLE
#   just launch_nnunet_train /path/to/data /path/to/output Dataset001_MitoLE "0" "48:00:00" "long"
# -------------------------------------------------------------------
default_nnunet_time := "48:00:00"
default_nnunet_partition := "long"

launch_nnunet_train data_folder output_dir dataset_id fold="0" time=default_nnunet_time partition=default_nnunet_partition:
    @echo "Submitting Slurm job for nnunet_train (dataset={{dataset_id}}, fold={{fold}}) to partition {{partition}}, time {{time}}"
    @mkdir -p logs
    python submit_slurm.py -e mitohard -t {{time}} -p {{partition}} --job_name "nnunet_Train_{{dataset_id}}_fold{{fold}}" --command "just nnunet_train {{data_folder}} {{output_dir}} {{dataset_id}} {{fold}}"
