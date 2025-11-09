"""
Configuration management using dataclasses and YAML.

Provides type-safe configuration management with support for:
- YAML file loading
- Command-line argument overriding
- Configuration validation
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    id: Union[int, str] = 1
    name: Optional[str] = None  # Auto-detect from id if None
    root: Optional[str] = None  # Use nnUNet_raw default if None


@dataclass
class TrainingConfig:
    """Training configuration."""
    iterations: int = 2000
    batch_size: int = 16
    learning_rate: float = 0.003
    patch_size: int = 256
    finetune_layer: str = "all"  # Options: none, stage4, stage3, stage2, stage1, all
    num_epochs: Optional[int] = None  # Alternative to iterations
    # MicroSAM-specific parameters
    n_epochs: Optional[int] = None  # For MicroSAM (alias for num_epochs)
    n_objects_per_batch: Optional[int] = None  # For MicroSAM
    train_instance_segmentation: Optional[bool] = None  # For MicroSAM


@dataclass
class ModelConfig:
    """Model configuration."""
    config_path: Optional[str] = None
    checkpoint: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None  # For MicroSAM


@dataclass
class InferenceConfig:
    """Inference configuration."""
    use_gpu: bool = True
    downsampling: int = 1
    confidence_thr: float = 0.5
    center_confidence_thr: float = 0.1
    min_distance_object_centers: int = 3
    min_size: int = 500
    min_extent: int = 5
    pixel_vote_thr: int = 2
    axes: List[str] = field(default_factory=lambda: ["xy", "xz", "yz"])
    tile_shape: Optional[List[int]] = None
    halo: Optional[List[int]] = None
    use_embeddings: bool = False
    label_path: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    log_dir: Optional[str] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    model_dir: str = "./checkpoints"
    log_dir: Optional[str] = None
    output_data_path: Optional[str] = None


@dataclass
class BaseConfig:
    """Base configuration class."""
    method: str = "mitonet"  # mitonet, microsam, nnunet
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        # Handle nested dataclasses
        if "dataset" in config_dict and isinstance(config_dict["dataset"], dict):
            config_dict["dataset"] = DatasetConfig(**config_dict["dataset"])
        elif "dataset" in config_dict and isinstance(config_dict["dataset"], (str, int)):
            # Handle legacy format where dataset is just an ID
            config_dict["dataset"] = DatasetConfig(id=config_dict["dataset"])

        if "model" in config_dict and isinstance(config_dict["model"], dict):
            config_dict["model"] = ModelConfig(**config_dict["model"])

        if "inference" in config_dict and isinstance(config_dict["inference"], dict):
            config_dict["inference"] = InferenceConfig(**config_dict["inference"])

        if "logging" in config_dict and isinstance(config_dict["logging"], dict):
            config_dict["logging"] = LoggingConfig(**config_dict["logging"])

        if "output" in config_dict and isinstance(config_dict["output"], dict):
            config_dict["output"] = OutputConfig(**config_dict["output"])

        # Handle training config if present
        if "training" in config_dict and isinstance(config_dict["training"], dict):
            # Merge training into inference for backward compatibility
            training = config_dict["training"]
            if "inference" not in config_dict:
                config_dict["inference"] = {}
            for key in ["iterations", "batch_size", "learning_rate", "patch_size"]:
                if key in training:
                    config_dict["inference"][key] = training[key]

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass
class MitoNetConfig(BaseConfig):
    """MitoNet-specific configuration."""
    method: str = "mitonet"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mode: str = "train"  # train or predict
    skip_conversion: bool = False
    test_after_training: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MitoNetConfig":
        """Create MitoNet config from dictionary."""
        # Handle legacy format
        if "dataset" in config_dict and isinstance(config_dict["dataset"], (str, int)):
            dataset_id = config_dict["dataset"]
            config_dict["dataset"] = {"id": dataset_id}

        if "training" in config_dict and isinstance(config_dict["training"], dict):
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        elif "iterations" in config_dict or "batch_size" in config_dict:
            # Legacy format: training params at top level
            training_dict = {}
            for key in ["iterations", "batch_size", "learning_rate", "patch_size", "finetune_layer"]:
                if key in config_dict:
                    training_dict[key] = config_dict.pop(key)
            config_dict["training"] = TrainingConfig(**training_dict)

        # Create base config first
        base_config = BaseConfig.from_dict(config_dict)
        
        # Create MitoNet-specific config - preserve dataclass objects
        # Don't use asdict() as it converts nested dataclasses to dicts
        mitonet_dict = {
            "method": "mitonet",
            "dataset": base_config.dataset,
            "model": base_config.model,
            "inference": base_config.inference,
            "logging": base_config.logging,
            "output": base_config.output,
        }
        
        # Add training config
        if hasattr(base_config, "training"):
            mitonet_dict["training"] = base_config.training
        elif "training" in config_dict:
            if isinstance(config_dict["training"], TrainingConfig):
                mitonet_dict["training"] = config_dict["training"]
            else:
                mitonet_dict["training"] = TrainingConfig(**config_dict["training"])
        else:
            mitonet_dict["training"] = TrainingConfig()
        
        # Add MitoNet-specific fields
        for key in ["mode", "skip_conversion", "test_after_training"]:
            if key in config_dict:
                mitonet_dict[key] = config_dict[key]

        return cls(**mitonet_dict)


@dataclass
class MicroSAMConfig(BaseConfig):
    """MicroSAM-specific configuration."""
    method: str = "microsam"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mode: str = "ZS"  # ZS (zero-shot) or FT (fine-tuning)
    eval: bool = True
    test_after_training: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MicroSAMConfig":
        """Create MicroSAM config from dictionary."""
        # Handle legacy format
        if "dataset" in config_dict and isinstance(config_dict["dataset"], (str, int)):
            dataset_id = config_dict["dataset"]
            config_dict["dataset"] = {"id": dataset_id}

        # Handle training config
        if "training" in config_dict and isinstance(config_dict["training"], dict):
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        # Create base config first
        base_config = BaseConfig.from_dict(config_dict)
        
        # Create MicroSAM-specific config - preserve dataclass objects
        microsam_dict = {
            "method": "microsam",
            "dataset": base_config.dataset,
            "model": base_config.model,
            "inference": base_config.inference,
            "logging": base_config.logging,
            "output": base_config.output,
        }
        
        # Add training config if present
        if hasattr(base_config, "training"):
            microsam_dict["training"] = base_config.training
        elif "training" in config_dict:
            if isinstance(config_dict["training"], TrainingConfig):
                microsam_dict["training"] = config_dict["training"]
            else:
                microsam_dict["training"] = TrainingConfig(**config_dict["training"])
        
        # Add MicroSAM-specific fields
        for key in ["mode", "eval", "test_after_training"]:
            if key in config_dict:
                microsam_dict[key] = config_dict[key]

        return cls(**microsam_dict)


@dataclass
class NNUNetPipelineConfig:
    """Pipeline options for nnUNet training."""
    output_dir: Optional[str] = None
    fold: int = 0
    trainer: str = "nnUNetTrainer"
    max_epochs: int = 1000
    skip_boundary: bool = False
    skip_plan: bool = False
    skip_training: bool = False
    skip_prediction: bool = False
    skip_postprocess: bool = False
    skip_evaluation: bool = False


@dataclass
class NNUNetConfig(BaseConfig):
    """nnUNet-specific configuration."""
    method: str = "nnunet"
    nnunet: NNUNetPipelineConfig = field(default_factory=NNUNetPipelineConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NNUNetConfig":
        """Create nnUNet config from dictionary."""
        # Ensure dataset structure matches DatasetConfig expectations
        if "dataset" in config_dict and isinstance(config_dict["dataset"], (str, int)):
            dataset_id = config_dict["dataset"]
            config_dict["dataset"] = {"id": dataset_id}

        # Extract nnunet pipeline config
        pipeline_cfg = config_dict.get("nnunet", {})
        if isinstance(pipeline_cfg, NNUNetPipelineConfig):
            nnunet_pipeline = pipeline_cfg
        else:
            nnunet_pipeline = NNUNetPipelineConfig(**pipeline_cfg) if isinstance(pipeline_cfg, dict) else NNUNetPipelineConfig()

        # Build base configuration (ignoring nnunet field which BaseConfig doesn't know about)
        base_config = BaseConfig.from_dict({k: v for k, v in config_dict.items() if k != "nnunet"})

        return cls(
            method="nnunet",
            dataset=base_config.dataset,
            model=base_config.model,
            inference=base_config.inference,
            logging=base_config.logging,
            output=base_config.output,
            nnunet=nnunet_pipeline,
        )


def load_config(config_path: Union[str, Path], method: Optional[str] = None) -> BaseConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.
        method: Method name (mitonet, microsam, nnunet). If None, auto-detect from config.

    Returns:
        Configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Auto-detect method if not specified
    if method is None:
        method = config_dict.get("method", "mitonet")

    # Create appropriate config class
    if method == "mitonet":
        return MitoNetConfig.from_dict(config_dict)
    elif method == "microsam":
        return MicroSAMConfig.from_dict(config_dict)
    elif method == "nnunet":
        return NNUNetConfig.from_dict(config_dict)
    else:
        return BaseConfig.from_dict(config_dict)


def save_config(config: BaseConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object.
        path: Path to save configuration file.
    """
    config.save(path)


def update_config_from_args(config: BaseConfig, args: Dict[str, Any]) -> BaseConfig:
    """
    Update configuration from command-line arguments.

    Args:
        config: Base configuration object.
        args: Dictionary of command-line arguments.

    Returns:
        Updated configuration object.
    """
    config_dict = config.to_dict()

    # Update dataset
    if "dataset" in args and args["dataset"] is not None:
        if isinstance(config.dataset, DatasetConfig):
            config.dataset.id = args["dataset"]
        else:
            config_dict["dataset"] = {"id": args["dataset"]}

    # Update other fields
    for key, value in args.items():
        if value is not None and key != "dataset":
            # Handle nested updates
            if "." in key:
                parts = key.split(".")
                if len(parts) == 2:
                    if parts[0] not in config_dict:
                        config_dict[parts[0]] = {}
                    config_dict[parts[0]][parts[1]] = value
            else:
                config_dict[key] = value

    # Recreate config from updated dict
    if isinstance(config, MitoNetConfig):
        return MitoNetConfig.from_dict(config_dict)
    elif isinstance(config, MicroSAMConfig):
        return MicroSAMConfig.from_dict(config_dict)
    elif isinstance(config, NNUNetConfig):
        return NNUNetConfig.from_dict(config_dict)
    else:
        return BaseConfig.from_dict(config_dict)
