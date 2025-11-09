"""
Dataset base classes for mitoem2.

Provides base classes for different dataset types used in the project.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base dataset class for all mitoem2 datasets.

    This class provides a common interface for different dataset types
    (nnUNet format, empanada format, etc.).
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing the dataset.
            split: Dataset split (train, val, test).
            transform: Optional transform to apply to samples.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> list:
        """
        Load list of sample paths.

        Returns:
            List of sample identifiers or paths.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing image and label tensors.
        """
        pass


class MitoNetDataset(BaseDataset):
    """
    Dataset for MitoNet training (empanada format).

    Expects data organized as:
        data_root/
            train/
                volume_name/
                    images/
                        slice_*.tif
                    masks/
                        slice_*.tif
            val/
                volume_name/
                    images/
                        slice_*.tif
                    masks/
                        slice_*.tif
    """

    def _load_samples(self) -> list:
        """Load list of sample paths."""
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        samples = []
        for volume_dir in sorted(split_dir.iterdir()):
            if not volume_dir.is_dir():
                continue

            images_dir = volume_dir / "images"
            masks_dir = volume_dir / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            # Get all image files
            image_files = sorted(images_dir.glob("*.tif*"))
            for img_file in image_files:
                mask_file = masks_dir / img_file.name
                if mask_file.exists():
                    samples.append({
                        "image": img_file,
                        "mask": mask_file,
                        "volume": volume_dir.name,
                    })

        return samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        # Load image and mask
        import tifffile
        image = tifffile.imread(sample["image"])
        mask = tifffile.imread(sample["mask"])

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "volume": sample["volume"],
            "slice": sample["image"].stem,
        }


class MicroSAMDataset(BaseDataset):
    """
    Dataset for MicroSAM training.

    Similar structure to MitoNetDataset but may have different preprocessing.
    """

    def _load_samples(self) -> list:
        """Load list of sample paths."""
        # For now, use same structure as MitoNetDataset
        # Can be customized later
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        samples = []
        for volume_dir in sorted(split_dir.iterdir()):
            if not volume_dir.is_dir():
                continue

            images_dir = volume_dir / "images"
            masks_dir = volume_dir / "masks"

            if not images_dir.exists() or not masks_dir.exists():
                continue

            image_files = sorted(images_dir.glob("*.tif*"))
            for img_file in image_files:
                mask_file = masks_dir / img_file.name
                if mask_file.exists():
                    samples.append({
                        "image": img_file,
                        "mask": mask_file,
                        "volume": volume_dir.name,
                    })

        return samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        # Load image and mask
        import tifffile
        image = tifffile.imread(sample["image"])
        mask = tifffile.imread(sample["mask"])

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "volume": sample["volume"],
            "slice": sample["image"].stem,
        }


class nnUNetDataset(BaseDataset):
    """
    Dataset for nnUNet format.

    Expects data organized as:
        data_root/
            imagesTr/
                *.nii.gz
            labelsTr/ (or instancesTr/)
                *.nii.gz
            imagesTs/
                *.nii.gz
            labelsTs/ (or instancesTs/)
                *.nii.gz
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform: Optional[Any] = None,
        use_instances: bool = True,
    ):
        """
        Initialize nnUNet dataset.

        Args:
            data_root: Root directory containing nnUNet dataset.
            split: Dataset split (train or test).
            transform: Optional transform to apply.
            use_instances: Whether to use instancesTr/instancesTs instead of labelsTr/labelsTs.
        """
        self.use_instances = use_instances
        super().__init__(data_root, split, transform)

    def _load_samples(self) -> list:
        """Load list of sample paths."""
        if self.split == "train":
            images_dir = self.data_root / "imagesTr"
            if self.use_instances:
                labels_dir = self.data_root / "instancesTr"
            else:
                labels_dir = self.data_root / "labelsTr"
        else:  # test
            images_dir = self.data_root / "imagesTs"
            if self.use_instances:
                labels_dir = self.data_root / "instancesTs"
            else:
                labels_dir = self.data_root / "labelsTs"

        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"Dataset directories not found in {self.data_root}")

        samples = []
        image_files = sorted(images_dir.glob("*.nii.gz"))
        for img_file in image_files:
            # Find corresponding label file
            base_name = img_file.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            label_file = labels_dir / f"{base_name}.nii.gz"
            
            if label_file.exists():
                samples.append({
                    "image": img_file,
                    "mask": label_file,
                })

        return samples

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        # Load 3D volumes
        import SimpleITK as sitk
        img_sitk = sitk.ReadImage(str(sample["image"]))
        mask_sitk = sitk.ReadImage(str(sample["mask"]))
        
        image = sitk.GetArrayFromImage(img_sitk)
        mask = sitk.GetArrayFromImage(mask_sitk)

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Apply transforms if provided
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "filename": sample["image"].stem,
        }
