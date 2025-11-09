"""
Data loader utilities for mitoem2.
"""
from torch.utils.data import DataLoader
from typing import Optional, Callable
from pathlib import Path

from mitoem2.data.dataset import BaseDataset, MitoNetDataset, MicroSAMDataset, nnUNetDataset


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader from a dataset.

    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_mitonet_dataloader(
    data_root: Path,
    split: str = "train",
    batch_size: int = 16,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for MitoNet training.

    Args:
        data_root: Root directory containing empanada-format data.
        split: Dataset split (train, val).
        batch_size: Batch size.
        transform: Optional transform to apply.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        DataLoader instance.
    """
    dataset = MitoNetDataset(data_root=data_root, split=split, transform=transform)
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_microsam_dataloader(
    data_root: Path,
    split: str = "train",
    batch_size: int = 1,
    transform: Optional[Callable] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for MicroSAM training.

    Args:
        data_root: Root directory containing data.
        split: Dataset split (train, val).
        batch_size: Batch size.
        transform: Optional transform to apply.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        DataLoader instance.
    """
    dataset = MicroSAMDataset(data_root=data_root, split=split, transform=transform)
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_nnunet_dataloader(
    data_root: Path,
    split: str = "train",
    batch_size: int = 1,
    transform: Optional[Callable] = None,
    use_instances: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for nnUNet format data.

    Args:
        data_root: Root directory containing nnUNet dataset.
        split: Dataset split (train, test).
        batch_size: Batch size.
        transform: Optional transform to apply.
        use_instances: Whether to use instances instead of labels.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        DataLoader instance.
    """
    dataset = nnUNetDataset(
        data_root=data_root,
        split=split,
        transform=transform,
        use_instances=use_instances,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
