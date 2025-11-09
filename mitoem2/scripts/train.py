#!/usr/bin/env python
"""
Unified training script for mitoem2.

Supports training for all three methods: MitoNet, MicroSAM, and nnUNet.
"""
import argparse
from pathlib import Path

from mitoem2.configs import load_config, NNUNetConfig
from mitoem2.utils.logging import setup_logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train mitoem2 models")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["mitonet", "microsam", "nnunet"],
        help="Method to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID or name (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for checkpoints (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    # nnUNet-specific options
    parser.add_argument(
        "--nnunet-output-dir",
        type=str,
        help="Base output directory for nnUNet pipeline (overrides config.nnunet.output_dir)",
    )
    parser.add_argument(
        "--nnunet-fold",
        type=int,
        help="Fold index for nnUNet training/prediction",
    )
    parser.add_argument(
        "--nnunet-trainer",
        type=str,
        help="nnUNet trainer class name (e.g., nnUNetTrainer)",
    )
    parser.add_argument(
        "--nnunet-max-epochs",
        type=int,
        help="Maximum epochs for nnUNet training",
    )
    parser.add_argument("--nnunet-skip-boundary", action="store_true", help="Skip boundary mask generation step")
    parser.add_argument("--nnunet-skip-plan", action="store_true", help="Skip nnUNet plan and preprocess step")
    parser.add_argument("--nnunet-skip-training", action="store_true", help="Skip nnUNet training step")
    parser.add_argument("--nnunet-skip-prediction", action="store_true", help="Skip nnUNet prediction step")
    parser.add_argument("--nnunet-skip-postprocess", action="store_true", help="Skip post-processing step")
    parser.add_argument("--nnunet-skip-evaluation", action="store_true", help="Skip evaluation step")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config, method=args.method)
    else:
        # Use default config based on method
        if args.method == "mitonet":
            config_path = Path(__file__).parent.parent / "configs" / "mitonet" / "train_default.yaml"
        elif args.method == "microsam":
            config_path = Path(__file__).parent.parent / "configs" / "microsam" / "train_default.yaml"
        elif args.method == "nnunet":
            config_path = Path(__file__).parent.parent / "configs" / "nnunet" / "train_default.yaml"
        else:
            raise ValueError(f"No default config for method: {args.method}")
        config = load_config(config_path, method=args.method)

    # For nnUNet, always derive output/model directories from nnunetv2.paths
    if args.method == "nnunet":
        try:
            from nnunetv2.paths import nnUNet_results
        except ImportError:
            pass
        else:
            if isinstance(config.output, dict):
                config.output["model_dir"] = nnUNet_results
            else:
                config.output.model_dir = nnUNet_results

    # Override with command-line arguments
    if args.dataset:
        # Ensure dataset is a DatasetConfig object, not a dict
        if isinstance(config.dataset, dict):
            from mitoem2.configs.config import DatasetConfig
            config.dataset = DatasetConfig(id=args.dataset)
        else:
            config.dataset.id = args.dataset
    if args.output_dir:
        # Ensure output is an OutputConfig object, not a dict
        if isinstance(config.output, dict):
            from mitoem2.configs.config import OutputConfig
            config.output = OutputConfig(model_dir=args.output_dir)
        else:
            config.output.model_dir = args.output_dir

    # Update nnUNet-specific configuration overrides
    if args.method == "nnunet" and isinstance(config, NNUNetConfig):
        pipeline_cfg = config.nnunet
        if args.nnunet_output_dir:
            pipeline_cfg.output_dir = args.nnunet_output_dir
        if args.nnunet_fold is not None:
            pipeline_cfg.fold = args.nnunet_fold
        if args.nnunet_trainer:
            pipeline_cfg.trainer = args.nnunet_trainer
        if args.nnunet_max_epochs is not None:
            pipeline_cfg.max_epochs = args.nnunet_max_epochs

        if args.nnunet_skip_boundary:
            pipeline_cfg.skip_boundary = True
        if args.nnunet_skip_plan:
            pipeline_cfg.skip_plan = True
        if args.nnunet_skip_training:
            pipeline_cfg.skip_training = True
        if args.nnunet_skip_prediction:
            pipeline_cfg.skip_prediction = True
        if args.nnunet_skip_postprocess:
            pipeline_cfg.skip_postprocess = True
        if args.nnunet_skip_evaluation:
            pipeline_cfg.skip_evaluation = True

    # Setup logging
    # Ensure config attributes are objects, not dicts
    output_dir = None
    if isinstance(config.output, dict):
        output_dir = config.output.get("model_dir")
    else:
        output_dir = getattr(config.output, "model_dir", None)
    
    use_wandb = False
    wandb_project = None
    if isinstance(config.logging, dict):
        use_wandb = config.logging.get("use_wandb", False)
        wandb_project = config.logging.get("wandb_project")
    else:
        use_wandb = getattr(config.logging, "use_wandb", False)
        wandb_project = getattr(config.logging, "wandb_project", None)
    
    logger = setup_logger(
        name=f"mitoem2_train_{args.method}",
        log_dir=Path(output_dir) if output_dir else None,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=config.to_dict(),
    )

    logger.info("Starting training for %s", args.method)
    try:
        logger.info("Configuration: %s", config.to_dict())
    except Exception as exc:
        logger.warning("Could not serialize config to dict: %s", exc)
        dataset_id_value = getattr(config.dataset, "id", None)
        if dataset_id_value is None and isinstance(config.dataset, dict):
            dataset_id_value = config.dataset.get("id")
        logger.info("Dataset ID: %s", dataset_id_value)

    # Train based on method
    if args.method == "mitonet":
        # Check if trainer module exists, otherwise show message
        #try:
        from mitoem2.training.mitonet_trainer import MitoNetTrainer
        trainer = MitoNetTrainer.from_config(config)
        # Get iterations from training config
        if hasattr(config, "training"):
            if isinstance(config.training, dict):
                iterations = config.training.get("iterations", 2000)
            else:
                iterations = getattr(config.training, "iterations", 2000)
        else:
            iterations = 2000
        trainer.train(
            num_epochs=iterations,
            checkpoint_path=Path(args.resume) if args.resume else None,
        )
        # except ImportError:
        #     logger.error("MitoNetTrainer not yet implemented. Please use the legacy training script:")
        #     logger.error("  python src/training/mitoNet_finetune.py -d 1 --mode train")
        #     return
    elif args.method == "microsam":
        #try:
        from mitoem2.training.microsam_trainer import MicroSAMTrainer
        trainer = MicroSAMTrainer.from_config(config)
        # Get n_epochs from training config
        if hasattr(config, "training"):
            if isinstance(config.training, dict):
                n_epochs = config.training.get("n_epochs", 10)
            else:
                n_epochs = getattr(config.training, "n_epochs", 10)
        else:
            n_epochs = 10
        trainer.train(
            num_epochs=n_epochs,
            checkpoint_path=Path(args.resume) if args.resume else None,
        )
        # except ImportError:
        #     logger.error("MicroSAMTrainer not yet implemented. Please use the legacy training script:")
        #     logger.error("  python src/training/micro_sam_finetune.py --dataset_path <path>")
        #     return
    elif args.method == "nnunet":
        from mitoem2.training.nnunet_trainer import NNUNetTrainer

        if not isinstance(config, NNUNetConfig):
            raise TypeError("Expected NNUNetConfig for nnunet method")

        trainer = NNUNetTrainer.from_config(config, logger=logger)
        trainer.run_pipeline()
        return
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
