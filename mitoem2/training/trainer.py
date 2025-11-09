"""
Base trainer class for mitoem2.

Provides a unified training interface for all models.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mitoem2.models.base import BaseModel
from mitoem2.training.callbacks import Callback, EarlyStopping, LearningRateScheduler, WandbCallback, CheckpointCallback
from mitoem2.utils.logging import get_logger
from mitoem2.utils.device import get_device
from mitoem2.utils.checkpoint import save_checkpoint, load_checkpoint

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """
    Base trainer class for all mitoem2 models.

    This class provides a unified training interface with support for:
    - Multi-GPU training
    - Early stopping
    - Learning rate scheduling
    - Wandb logging
    - Checkpoint management
    """

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        use_multi_gpu: bool = False,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            optimizer: Optimizer.
            criterion: Loss function.
            device: Device to train on.
            use_multi_gpu: Whether to use multiple GPUs.
            callbacks: List of callbacks.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_multi_gpu = use_multi_gpu
        self.callbacks = callbacks or []

        # Setup device
        if device is None:
            device = get_device(use_gpu=True)
        self.device = device
        self.model = self.model.to(self.device)

        # Multi-GPU setup
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)

        # Training state
        self.current_epoch = 0
        self.should_stop = False
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data.

        Returns:
            Dictionary of metrics for this step.
        """
        pass

    @abstractmethod
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.

        Args:
            batch: Batch of data.

        Returns:
            Dictionary of metrics for this step.
        """
        pass

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Training step
            step_metrics = self.train_step(batch)
            loss = step_metrics.get("loss", 0.0)

            total_loss += loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            # Call batch callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self, loss)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Validation step
                step_metrics = self.validate_step(batch)
                loss = step_metrics.get("loss", 0.0)

                total_loss += loss
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"val_loss": avg_loss}

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train.
            start_epoch: Starting epoch (for resuming training).
            checkpoint_path: Path to checkpoint to resume from.
        """
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(
                checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                device=self.device,
            )
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(f"Resuming training from epoch {start_epoch}")

        self.current_epoch = start_epoch

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Call epoch start callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)

            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["train_loss"])

            # Validation
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics["val_loss"])

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(metrics)

            # Log metrics
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            # Call epoch end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, metrics)

            # Check if should stop
            if self.should_stop:
                logger.info("Early stopping triggered. Stopping training.")
                break

        logger.info("Training complete!")

    def save_checkpoint(
        self,
        checkpoint_dir: Path,
        checkpoint_name: str = "checkpoint",
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint.
            checkpoint_name: Name of checkpoint.
            is_best: Whether this is the best model.

        Returns:
            Path to saved checkpoint.
        """
        # Get model state dict (handle DataParallel)
        model_state_dict = self.model.state_dict()
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()

        return save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            loss=self.val_losses[-1] if self.val_losses else None,
            metrics=self.metrics_history[-1] if self.metrics_history else None,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            is_best=is_best,
        )
