import tifffile as tiff
import numpy as np

def bbox_3d_binary(mask: np.ndarray):
    """
    Compute tight 3D bounding box for a binary mask.
    Returns (zmin, zmax, ymin, ymax, xmin, xmax) where zmax/ymax/xmax are exclusive.
    """
    # mask: shape (Z, Y, X), values {0,1} or {False,True}
    if not np.any(mask):
        return None  # all background

    coords = np.nonzero(mask)              # tuple of arrays (z_idx, y_idx, x_idx)
    zmin, ymin, xmin = [c.min() for c in coords]
    zmax, ymax, xmax = [c.max() + 1 for c in coords]   # make max exclusive
    return (zmin, zmax, ymin, ymax, xmin, xmax)


if __name__ == "__main__":
    mask = tiff.imread("/projects/weilab/liupeng/dataset/mito/wei20/mitoEM-H_den_mito_downsampled.tiff")
    bbox = bbox_3d_binary(mask)
    print(f"Mask shape: {mask.shape}")
    print(f"Bounding box: {bbox}")