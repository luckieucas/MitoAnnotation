import os
import cv2
import torch
import numpy as np
from skimage import io
from pathlib import Path
import tifffile as tiff
from mitoem2.empanada.data._base import _BaseDataset
from mitoem2.empanada.data.utils import heatmap_and_offsets

__all__ = [
    'SingleClassInstanceDataset'
]

class SingleClassInstanceDataset(_BaseDataset):
    r"""Dataset for panoptic deeplab that supports a single instance
    class only.

    Args:
        data_dir: Str. Directory containing image/mask pairs. Structure should
        be data_dir -> source_datasets -> images/masks.

        transforms: Albumentations transforms to apply to images and masks.

        heatmap_sigma: Float. The standard deviation used for the gaussian
        blurring filter when converting object centers to a heatmap.

        weight_gamma: Float (0-1). Parameter than controls sampling of images
        within different source_datasets based on the number of images
        that that directory contains. Default is 0.3.

    """
    def __init__(
        self,
        data_dir,
        transforms=None,
        heatmap_sigma=6,
        weight_gamma=0.3,
    ):
        super(SingleClassInstanceDataset, self).__init__(
            data_dir, transforms, weight_gamma
        )
        self.heatmap_sigma = heatmap_sigma

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        mask_path = self.mskpaths[idx]
        
        # Read image - handle both .tif and other formats
        f_path = Path(f)
        if f_path.suffix.lower() in ['.tif', '.tiff']:
            image = tiff.imread(f)
        else:
            image = cv2.imread(f, 0)
        
        # Read mask - handle both .tif and other formats
        mask_path_obj = Path(mask_path)
        if mask_path_obj.suffix.lower() in ['.tif', '.tiff']:
            mask = tiff.imread(mask_path)
        else:
            mask = io.imread(mask_path)
        
        # Check if image was loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image: {f}. File may not exist or be corrupted.")
        
        # Ensure image is 2D or 3D numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image at {f} is not a numpy array. Got type: {type(image)}")
        
        # Handle grayscale images - add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]
        elif image.ndim == 3 and image.shape[-1] > 3:
            # If image has more than 3 channels, take first channel
            image = image[..., 0:1]
        mask = mask.astype('int32')
        if self.transforms is not None:
            output = self.transforms(image=image, mask=mask)
        else:
            output = {'image': image, 'mask': mask}

        mask = output['mask']
        heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
        output['ctr_hmp'] = heatmap
        output['offsets'] = offsets
        output['fname'] = f

        # the last step is to binarize the mask for semantic segmentation
        if isinstance(mask, torch.Tensor):
            output['sem'] = (mask.float() > 0).float()
        elif isinstance(mask, np.ndarray):
            output['sem'] = (mask.float() > 0).astype(np.float32)
        else:
            raise Exception(f'Invalid mask type {type(mask)}. Expect tensor or ndarray.')

        return output
