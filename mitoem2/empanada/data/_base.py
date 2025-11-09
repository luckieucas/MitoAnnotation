import os
import json
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from copy import deepcopy
import natsort

__all__ = [
    'BaseDataset'
]

class _BaseDataset(Dataset):
    r"""Pytorch Dataset class that supports addition of multiple
    segmentation datasets and computes sampling weights based
    on the count of images within subdirectories and a weight_gamma
    factor.
    """
    def __init__(
        self,
        data_dir,
        transforms=None,
        weight_gamma=None
    ):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir

        self.subdirs = []
        for sd in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, sd)):
                self.subdirs.append(sd)

        # images and masks as dicts ordered by subdirectory
        self.impaths_dict = {}
        self.mskpaths_dict = {}

        # Supported image extensions
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        for sd in self.subdirs:
            # Collect all image files
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(data_dir, f'{sd}/images/{ext}')))
            
            # Collect all mask files
            mask_files = []
            for ext in image_extensions:
                mask_files.extend(glob(os.path.join(data_dir, f'{sd}/masks/{ext}')))
            
            # Match image and mask files by filename (without path)
            # Only keep pairs where both image and mask exist
            image_files_sorted = natsort.natsorted(image_files)
            mask_files_sorted = natsort.natsorted(mask_files)
            
            # Create dictionaries keyed by filename
            image_dict = {}
            for img_path in image_files_sorted:
                img_name = os.path.basename(img_path)
                image_dict[img_name] = img_path
            
            mask_dict = {}
            for msk_path in mask_files_sorted:
                msk_name = os.path.basename(msk_path)
                mask_dict[msk_name] = msk_path
            
            # Find matching pairs
            matched_images = []
            matched_masks = []
            for img_name in natsort.natsorted(image_dict.keys()):
                if img_name in mask_dict:
                    matched_images.append(image_dict[img_name])
                    matched_masks.append(mask_dict[img_name])
                else:
                    import warnings
                    warnings.warn(f"Image {img_name} in {sd} has no corresponding mask, skipping.")
            
            # Warn about masks without images
            for msk_name in mask_dict.keys():
                if msk_name not in image_dict:
                    import warnings
                    warnings.warn(f"Mask {msk_name} in {sd} has no corresponding image, skipping.")
            
            self.impaths_dict[sd] = matched_images
            self.mskpaths_dict[sd] = matched_masks
            
            # Verify that we found files
            if len(self.impaths_dict[sd]) == 0:
                import warnings
                warnings.warn(f"No matching image/mask pairs found in {os.path.join(data_dir, f'{sd}/')}")

        # calculate weights per example, if weight gamma is not None
        self.weight_gamma = weight_gamma
        if weight_gamma is not None:
            self.weights = self._example_weights(self.impaths_dict, gamma=weight_gamma)
        else:
            self.weights = None

        # unpack dicts to lists of images - maintain order within each subdirectory
        self.impaths = []
        self.mskpaths = []
        for sd in self.subdirs:
            self.impaths.extend(self.impaths_dict[sd])
            self.mskpaths.extend(self.mskpaths_dict[sd])
        
        # Final verification that lengths match
        if len(self.impaths) != len(self.mskpaths):
            raise ValueError(
                f"Image and mask path counts do not match after matching: "
                f"{len(self.impaths)} images vs {len(self.mskpaths)} masks"
            )

        print(f'Found {len(self.subdirs)} image subdirectories with {len(self.impaths)} images.')

        self.transforms = transforms

    def __len__(self):
        return len(self.impaths)

    def __add__(self, add_dataset):
        # make a copy of self
        merged_dataset = deepcopy(self)

        # add the dicts and append lists/dicts
        for sd in add_dataset.impaths_dict.keys():
            if sd in merged_dataset.impaths_dict:
                # concat lists of paths together
                merged_dataset.impaths_dict[sd] += add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] += add_dataset.mskpaths_dict[sd]
            else:
                merged_dataset.impaths_dict[sd] = add_dataset.impaths_dict[sd]
                merged_dataset.mskpaths_dict[sd] = add_dataset.mskpaths_dict[sd]
        
        # Update subdirs to include all subdirectories
        merged_dataset.subdirs = list(set(merged_dataset.subdirs + list(add_dataset.impaths_dict.keys())))

        # recalculate weights
        if merged_dataset.weight_gamma is not None:
            merged_dataset.weights = self._example_weights(
                merged_dataset.impaths_dict, gamma=merged_dataset.weight_gamma
            )
        else:
            merged_dataset.weights = None

        # unpack dicts to lists of images - maintain order within each subdirectory
        merged_dataset.impaths = []
        merged_dataset.mskpaths = []
        for sd in merged_dataset.subdirs:
            if sd in merged_dataset.impaths_dict:
                merged_dataset.impaths.extend(merged_dataset.impaths_dict[sd])
                merged_dataset.mskpaths.extend(merged_dataset.mskpaths_dict[sd])
        
        # Final verification that lengths match
        if len(merged_dataset.impaths) != len(merged_dataset.mskpaths):
            raise ValueError(
                f"Image and mask path counts do not match after merging: "
                f"{len(merged_dataset.impaths)} images vs {len(merged_dataset.mskpaths)} masks"
            )

        return merged_dataset

    @staticmethod
    def _example_weights(paths_dict, gamma=0.3):
        # counts by source subdirectory
        counts = np.array(
            [len(paths) for paths in paths_dict.values()]
        )

        # invert and gamma the distribution
        weights = (1 / counts)
        weights = weights ** (gamma)

        # for interpretation, normalize weights
        # s.t. they sum to 1
        total_weights = weights.sum()
        weights /= total_weights

        # repeat weights per n images
        example_weights = []
        for w,c in zip(weights, counts):
            example_weights.extend([w] * c)

        return torch.tensor(example_weights)

    def __getitem__(self, idx):
        raise NotImplementedError
