"""
Data augmentation and transformation utilities.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Callable
import torchvision.transforms as transforms


class NormalizeImage:
    """Normalize image to [0, 1] range."""

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize image."""
        if image.max() > 1.0:
            image = image / 255.0
        return image, mask


class RandomFlip:
    """Random horizontal and/or vertical flip."""

    def __init__(self, p: float = 0.5):
        """
        Initialize random flip transform.

        Args:
            p: Probability of applying flip.
        """
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random flip."""
        if np.random.random() < self.p:
            # Horizontal flip
            if np.random.random() < 0.5:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            # Vertical flip
            if np.random.random() < 0.5:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
        return image, mask


class RandomRotate90:
    """Random 90-degree rotation."""

    def __init__(self, p: float = 0.5):
        """
        Initialize random rotation transform.

        Args:
            p: Probability of applying rotation.
        """
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random 90-degree rotation."""
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 1, 2, or 3 (90, 180, or 270 degrees)
            image = torch.rot90(image, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])
        return image, mask


class RandomCrop:
    """Random crop to specified size."""

    def __init__(self, size: int):
        """
        Initialize random crop transform.

        Args:
            size: Crop size (square crop).
        """
        self.size = size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop."""
        h, w = image.shape[-2:]
        if h < self.size or w < self.size:
            # Pad if image is too small
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
            mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
            h, w = image.shape[-2:]

        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)

        image = image[..., top:top+self.size, left:left+self.size]
        mask = mask[..., top:top+self.size, left:left+self.size]

        return image, mask


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: list):
        """
        Initialize compose transform.

        Args:
            transforms: List of transform functions.
        """
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


def get_train_transforms(patch_size: int = 256) -> Compose:
    """
    Get training transforms.

    Args:
        patch_size: Patch size for random crop.

    Returns:
        Composed transform.
    """
    return Compose([
        NormalizeImage(),
        RandomCrop(patch_size),
        RandomFlip(p=0.5),
        RandomRotate90(p=0.5),
    ])


def get_val_transforms() -> Compose:
    """
    Get validation transforms (minimal augmentation).

    Returns:
        Composed transform.
    """
    return Compose([
        NormalizeImage(),
    ])
