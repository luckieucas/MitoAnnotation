import os
import argparse
import numpy as np
import nibabel as nib
import tifffile as tiff
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.transform import resize
import cc3d
from connectomics.data.utils import getSegType

def cast2dtype(segm):
    """Cast segmentation mask to optimal dtype."""
    max_id = int(segm.max())
    dtype = getSegType(max_id)
    return segm.astype(dtype)

def bc_watershed(semantic,
                  boundary,
                  thres1=0.95,
                  thres2=0.4,
                  thres3=0.9,
                  thres_small=128,
                  scale_factors=(1.0, 1.0, 1.0),
                  seed_thres=20):
    # 前景与种子生成
    fg = semantic > thres3
    seeds = (semantic > thres1) & (boundary < thres2)
    lbl = label(seeds)
    print(f"seed count: {lbl.max()}")
    lbl = remove_small_objects(lbl, seed_thres)

    # Watershed 分割
    seg = watershed(-semantic.astype(np.float64), lbl, mask=fg)
    seg = remove_small_objects(seg, thres_small)

    # 可选缩放
    if not all(sf == 1.0 for sf in scale_factors):
        new_shape = (
            int(seg.shape[0] * scale_factors[0]),
            int(seg.shape[1] * scale_factors[1]),
            int(seg.shape[2] * scale_factors[2])
        )
        seg = resize(seg, new_shape, order=0,
                     preserve_range=True, anti_aliasing=False)

    # cast to optimal dtype before cc3d
    seg = cast2dtype(seg)

    # 3D 连通域分析，拆分非连通块
    seg = cc3d.connected_components(seg, connectivity=6)
    seg = cast2dtype(seg)

    return seg

def process_folder(input_folder, output_folder, save_tiff=False, save_nii=False):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith('.npz'):
            continue

        base, _ = os.path.splitext(fname)
        npz_path = os.path.join(input_folder, fname)
        ref_nii = os.path.join(input_folder, base + '.nii.gz')
        ref_tiff = os.path.join(input_folder, base + '.tiff')

        # 加载参考 affine 和 shape
        if os.path.exists(ref_nii):
            ref_img = nib.load(ref_nii)
            affine = ref_img.affine
            ref_shape = ref_img.shape
        elif os.path.exists(ref_tiff):
            ref_img = tiff.imread(ref_tiff)
            affine = np.eye(4)
            ref_shape = ref_img.shape
        else:
            print(f"[Warning] no reference NIfTI for {fname}, using identity affine")
            affine = np.eye(4)
            ref_shape = None

        # 加载 semantic 和 boundary
        data = np.load(npz_path)
        arr = data[data.files[0]]  # [3, D, H, W]
        semantic, boundary = arr[1], arr[2]
        print(f"{fname}: semantic shape {semantic.shape}")

        # 计算分割
        seg = bc_watershed(semantic, boundary, thres1=0.95, thres2=0.5, thres3=0.6, thres_small=64, scale_factors=(1.0, 1.0, 1.0), seed_thres=20)

        # 重排 axes Z,Y,X → X,Y,Z 用于 NIfTI
        if save_nii:
            seg_t = seg.transpose(2, 1, 0)
            if ref_shape and seg_t.shape != tuple(ref_shape):
                raise RuntimeError(f"Reordered seg shape {seg_t.shape} != ref shape {ref_shape}")

            # 保存 NIfTI
            nii = nib.Nifti1Image(seg_t, affine)
            out_nii = os.path.join(output_folder, base + '_seg.nii.gz')
            nib.save(nii, out_nii)
            print(f"Saved NIfTI: {out_nii}")

        # 可选：保存 TIFF 体数据
        if save_tiff:
            out_tiff = os.path.join(output_folder, base + '_seg.tiff')
            # 保存原始 seg (Z, Y, X)
            tiff.imwrite(out_tiff, seg, compression='zlib')
            print(f"Saved TIFF: {out_tiff}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True,
                   help="Folder with .npz volumes + matching .nii.gz")
    p.add_argument('-o', '--output', required=True,
                   help="Folder to save _seg.nii.gz outputs")
    p.add_argument('--save-tiff', action='store_true',
                   help="Also save segmentation result as a TIFF volume")
    args = p.parse_args()
    process_folder(args.input, args.output, save_tiff=args.save_tiff)
