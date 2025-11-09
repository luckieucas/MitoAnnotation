"""
Logging utilities for mitoem2.

Provides centralized logging configuration with support for file and console logging,
and optional integration with wandb.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logger(
    name: str = "mitoem2",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, uses default log directory.
        log_file: Log file name. If None, uses timestamp-based name.
        use_wandb: Whether to initialize wandb logging.
        wandb_project: Wandb project name.
        wandb_config: Wandb configuration dictionary.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is None:
        from mitoem2.utils.paths import get_log_dir
        log_dir = get_log_dir()
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Wandb initialization
    if use_wandb and WANDB_AVAILABLE:
        if wandb_project is None:
            wandb_project = "mitoem2"
        
        wandb.init(
            project=wandb_project,
            config=wandb_config or {},
            name=name,
        )
        logger.info(f"Wandb initialized with project: {wandb_project}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("Wandb requested but not available. Install with: pip install wandb")

    return logger


def get_logger(name: str = "mitoem2") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # If no handlers, set up with defaults
        return setup_logger(name=name)
    return logger
