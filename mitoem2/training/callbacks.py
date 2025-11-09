"""
Training callbacks for mitoem2.

Provides callbacks for early stopping, learning rate scheduling, and wandb logging.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from pathlib import Path

from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class Callback(ABC):
    """Base callback class."""

    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_end(self, batch: int, trainer: Any, loss: float) -> None:
        """Called at the end of each batch (optional)."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor.
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' or 'max' for the monitored metric.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        if self.monitor not in metrics:
            logger.warning(f"Metric {self.monitor} not found in metrics. Skipping early stopping.")
            return

        current_value = metrics[self.monitor]

        if self.mode == "min":
            is_better = current_value < (self.best_value - self.min_delta)
        else:
            is_better = current_value > (self.best_value + self.min_delta)

        if is_better:
            self.best_value = current_value
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict().copy()
            logger.info(f"Early stopping: {self.monitor} improved to {current_value:.4f}")
        else:
            self.wait += 1
            logger.info(
                f"Early stopping: {self.monitor} did not improve. "
                f"Wait: {self.wait}/{self.patience}"
            )

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                trainer.model.load_state_dict(self.best_weights)
                logger.info("Restored best weights from early stopping.")


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = "val_loss",
    ):
        """
        Initialize learning rate scheduler.

        Args:
            scheduler: PyTorch learning rate scheduler.
            monitor: Metric to monitor (for ReduceLROnPlateau).
        """
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
            else:
                logger.warning(f"Metric {self.monitor} not found for ReduceLROnPlateau scheduler.")
        else:
            self.scheduler.step()

        current_lr = trainer.optimizer.param_groups[0]["lr"]
        logger.info(f"Learning rate: {current_lr:.6f}")


class WandbCallback(Callback):
    """Wandb logging callback."""

    def __init__(self, project: str = "mitoem2", config: Optional[Dict[str, Any]] = None):
        """
        Initialize wandb callback.

        Args:
            project: Wandb project name.
            config: Configuration dictionary to log.
        """
        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            self.wandb_available = False
            logger.warning("wandb not available. Install with: pip install wandb")

        self.project = project
        self.config = config or {}
        self.initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure wandb is initialized."""
        if not self.wandb_available:
            return

        if not self.initialized:
            self.wandb.init(project=self.project, config=self.config)
            self.initialized = True

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        self._ensure_initialized()

    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        if not self.wandb_available:
            return

        self._ensure_initialized()
        self.wandb.log(metrics, step=epoch)

    def on_batch_end(self, batch: int, trainer: Any, loss: float) -> None:
        """Called at the end of each batch."""
        if not self.wandb_available:
            return

        self._ensure_initialized()
        self.wandb.log({"batch_loss": loss}, commit=False)


class CheckpointCallback(Callback):
    """Checkpoint saving callback."""

    def __init__(
        self,
        checkpoint_dir: Path,
        save_freq: int = 1,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            save_freq: Frequency of checkpoint saving (every N epochs).
            save_best: Whether to save best model.
            monitor: Metric to monitor for best model.
            mode: 'min' or 'max' for the monitored metric.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

        from mitoem2.utils.checkpoint import save_checkpoint
        self.save_checkpoint = save_checkpoint

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, trainer: Any, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        # Save regular checkpoint
        if epoch % self.save_freq == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            self.save_checkpoint(
                model=trainer.model,
                optimizer=trainer.optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_name=f"checkpoint_epoch_{epoch}",
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]
            is_best = (
                (current_value < self.best_value) if self.mode == "min"
                else (current_value > self.best_value)
            )

            if is_best:
                self.best_value = current_value
                checkpoint_path = self.checkpoint_dir / "checkpoint_best.pth"
                self.save_checkpoint(
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    epoch=epoch,
                    loss=metrics.get("val_loss", None),
                    metrics=metrics,
                    checkpoint_dir=self.checkpoint_dir,
                    checkpoint_name="checkpoint_best",
                    is_best=True,
                )
                logger.info(f"Saved best model: {checkpoint_path}")
