"""Training modules for mitoem2."""
from mitoem2.training.trainer import BaseTrainer
from mitoem2.training.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    WandbCallback,
    CheckpointCallback,
)

# Try to import method-specific trainers
try:
    from mitoem2.training.mitonet_trainer import MitoNetTrainer
except ImportError:
    MitoNetTrainer = None

try:
    from mitoem2.training.microsam_trainer import MicroSAMTrainer
except ImportError:
    MicroSAMTrainer = None

try:
    from mitoem2.training.nnunet_trainer import NNUNetTrainer
except ImportError:
    NNUNetTrainer = None

__all__ = [
    "BaseTrainer",
    "Callback",
    "EarlyStopping",
    "LearningRateScheduler",
    "WandbCallback",
    "CheckpointCallback",
    "MitoNetTrainer",
    "MicroSAMTrainer",
    "NNUNetTrainer",
]