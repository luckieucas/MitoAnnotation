import os
import argparse
import numpy as np
import tifffile as tiff
from glob import glob

import torch

import torch_em
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data


DATA_FOLDER = "data"


def convert_3d_to_2d_slices(nnunet_dataset_path, output_path):
    """
    将nnUNet数据集的3D tiff文件转换为2D slices
    
    Args:
        nnunet_dataset_path: nnUNet数据集路径，包含imagesTr, imagesTs, labelsTr, instancesTs等文件夹
        output_path: 输出路径，将创建train和val文件夹
    """
    print(f"Converting nnUNet dataset from {nnunet_dataset_path} to 2D slices...")
    
    # 处理训练集
    print("Processing training data...")
    imagesTr_path = os.path.join(nnunet_dataset_path, 'imagesTr')
    labelsTr_path = os.path.join(nnunet_dataset_path, 'instancesTr')

    train_images = sorted(glob(os.path.join(imagesTr_path, '*.tif*')))
    train_labels = sorted(glob(os.path.join(labelsTr_path, '*.tif*')))
    
    print(f"Found {len(train_images)} training images and {len(train_labels)} training labels")
    
    for img_path in train_images:
        # 读取3D图像
        img_3d = tiff.imread(img_path)
        base_name = os.path.basename(img_path).replace('_0000', '').replace('.tiff', '').replace('.tif', '')
        
        # 找到对应的标签文件
        label_path = None
        for lbl_path in train_labels:
            if base_name in os.path.basename(lbl_path):
                label_path = lbl_path
                break
        
        if label_path is None:
            print(f"Warning: No label found for {base_name}, skipping...")
            continue
        
        label_3d = tiff.imread(label_path)
        
        # 确保图像和标签的形状匹配
        if img_3d.shape != label_3d.shape:
            print(f"Warning: Shape mismatch for {base_name}: img {img_3d.shape} vs label {label_3d.shape}, skipping...")
            continue
        
        # 为每个volume创建子目录
        volume_train_dir = os.path.join(output_path, 'train', base_name)
        volume_images_dir = os.path.join(volume_train_dir, 'images')
        volume_masks_dir = os.path.join(volume_train_dir, 'masks')
        os.makedirs(volume_images_dir, exist_ok=True)
        os.makedirs(volume_masks_dir, exist_ok=True)
        
        # 将每个z切片保存为单独的2D tiff
        num_slices = img_3d.shape[0]
        print(f"  Converting {base_name}: {num_slices} slices")
        
        saved_slices = 0
        skipped_slices = 0
        
        for z in range(num_slices):
            slice_name = f"slice_{z:04d}.tif"
            
            # 检查标签切片是否全是背景
            label_slice = label_3d[z]
            if np.all(label_slice == 0):
                skipped_slices += 1
                continue  # 跳过全背景的切片
            
            # 保存图像切片
            img_slice = img_3d[z]
            # 确保图像是uint8类型
            if img_slice.dtype != np.uint8:
                if img_slice.max() <= 255:
                    img_slice = img_slice.astype(np.uint8)
                else:
                    # 归一化到0-255
                    img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(volume_images_dir, slice_name), img_slice, compression='zlib')
            
            # 保存标签切片
            # 确保标签是uint16类型
            label_slice = label_slice.astype(np.uint16)
            tiff.imwrite(os.path.join(volume_masks_dir, slice_name), label_slice, compression='zlib')
            
            saved_slices += 1
        
        print(f"    Saved: {saved_slices} slices, Skipped: {skipped_slices} empty slices")
    
    # 处理测试集（用作验证集）
    print("Processing test/validation data...")
    imagesTs_path = os.path.join(nnunet_dataset_path, 'imagesTs')
    instancesTs_path = os.path.join(nnunet_dataset_path, 'instancesTs')
    
    test_images = sorted(glob(os.path.join(imagesTs_path, '*.tif*')))
    test_labels = sorted(glob(os.path.join(instancesTs_path, '*.tif*')))
    
    print(f"Found {len(test_images)} test images and {len(test_labels)} test labels")
    
    for img_path in test_images:
        # 读取3D图像
        img_3d = tiff.imread(img_path)
        base_name = os.path.basename(img_path).replace('_0000', '').replace('.tiff', '').replace('.tif', '')
        
        # 找到对应的标签文件
        label_path = None
        for lbl_path in test_labels:
            if base_name in os.path.basename(lbl_path):
                label_path = lbl_path
                break
        
        if label_path is None:
            print(f"Warning: No label found for {base_name}, skipping...")
            continue
        
        label_3d = tiff.imread(label_path)
        
        # 确保图像和标签的形状匹配
        if img_3d.shape != label_3d.shape:
            print(f"Warning: Shape mismatch for {base_name}: img {img_3d.shape} vs label {label_3d.shape}, skipping...")
            continue
        
        # 为每个volume创建子目录
        volume_val_dir = os.path.join(output_path, 'val', base_name)
        volume_images_dir = os.path.join(volume_val_dir, 'images')
        volume_labels_dir = os.path.join(volume_val_dir, 'masks')
        os.makedirs(volume_images_dir, exist_ok=True)
        os.makedirs(volume_labels_dir, exist_ok=True)
        
        # 将每个z切片保存为单独的2D tiff
        num_slices = img_3d.shape[0]
        print(f"  Converting {base_name}: {num_slices} slices")
        
        saved_slices = 0
        skipped_slices = 0
        
        for z in range(num_slices):
            slice_name = f"slice_{z:04d}.tif"
            
            # 检查标签切片是否全是背景
            label_slice = label_3d[z]
            if np.all(label_slice == 0):
                skipped_slices += 1
                continue  # 跳过全背景的切片
            
            # 保存图像切片
            img_slice = img_3d[z]
            # 确保图像是uint8类型
            if img_slice.dtype != np.uint8:
                if img_slice.max() <= 255:
                    img_slice = img_slice.astype(np.uint8)
                else:
                    # 归一化到0-255
                    img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
            tiff.imwrite(os.path.join(volume_images_dir, slice_name), img_slice, compression='zlib')
            
            # 保存标签切片
            # 确保标签是int32类型
            if label_slice.dtype == np.uint16:
                label_slice = label_slice.astype(np.int32)
            tiff.imwrite(os.path.join(volume_labels_dir, slice_name), label_slice, compression='zlib')
            
            saved_slices += 1
        
        print(f"    Saved: {saved_slices} slices, Skipped: {skipped_slices} empty slices")
    
    print(f"Conversion complete! Data saved to {output_path}")
    # 使用递归glob统计文件数
    print(f"  Train images: {len(glob(os.path.join(output_path, 'train', '**/images/*.tif*'), recursive=True))}")
    print(f"  Train masks: {len(glob(os.path.join(output_path, 'train', '**/masks/*.tif*'), recursive=True))}")
    print(f"  Val images: {len(glob(os.path.join(output_path, 'val', '**/images/*.tif*'), recursive=True))}")
    print(f"  Val labels: {len(glob(os.path.join(output_path, 'val', '**/masks/*.tif*'), recursive=True))}")


def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation, nnunet_dataset_path):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the example hela data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    
    Args:
        split: "train" or "val"
        patch_shape: Shape of patches for training
        batch_size: Batch size
        train_instance_segmentation: Whether to train instance segmentation
        nnunet_dataset_path: Path to the nnUNet dataset root directory
    """
    assert split in ("train", "val")
    
    # 检查是否存在2d_slices文件夹
    slices_path = os.path.join(nnunet_dataset_path, '2d_slices')
    
    if not os.path.exists(slices_path):
        print(f"2d_slices folder not found at {slices_path}")
        print("Creating 2D slices from 3D data...")
        convert_3d_to_2d_slices(nnunet_dataset_path, slices_path)
    else:
        print(f"Found existing 2d_slices folder at {slices_path}")
    
    # 设置图像和标签目录
    if split == "train":
        split_dir = os.path.join(slices_path, 'train')
    else:
        split_dir = os.path.join(slices_path, 'val')
    
    # 查找所有volume的images和masks目录
    image_dirs = sorted(glob(os.path.join(split_dir, '*/images')))
    mask_dirs = sorted(glob(os.path.join(split_dir, '*/masks')))
    
    if not image_dirs:
        raise ValueError(f"No image directories found in {split_dir}")
    if not mask_dirs:
        raise ValueError(f"No mask directories found in {split_dir}")
    
    print(f"Found {len(image_dirs)} volume(s) for {split} split:")
    for img_dir in image_dirs:
        volume_name = os.path.basename(os.path.dirname(img_dir))
        print(f"  - {volume_name}")
    
    # 使用第一个volume作为默认（如果有多个volume，torch_em可以处理多个路径）
    # 如果需要使用所有volumes，可以传递列表
    image_dir = image_dirs if len(image_dirs) > 1 else image_dirs[0]
    segmentation_dir = mask_dirs if len(mask_dirs) > 1 else mask_dirs[0]
    
    print(f"Using image directory: {image_dir}")
    print(f"Using segmentation directory: {segmentation_dir}")

    # 'torch_em.default_segmentation_loader' is a convenience function to build a torch dataloader
    # from image data and labels for training segmentation models.
    # It supports image data in various formats. Here, we load image data and labels from the two
    # folders with tif images that were downloaded by the example data functionality, by specifying
    # `raw_key` and `label_key` as `*.tif`. This means all images in the respective folders that end with
    # .tif will be loaded.
    # The function supports many other file formats. For example, if you have tif stacks with multiple slices
    # instead of multiple tif images in a foldder, then you can pass raw_key=label_key=None.

    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = "*.tif", "*.tif"
    # Alternative: if you have tif stacks you can just set raw_key and label_key to None
    # raw_key, label_key= None, None

    # No need for ROI since we already split data into train/val folders
    roi = None

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components
    
    # Add a sampler to reject patches without foreground objects
    # This prevents "No foreground objects were found" errors during training
    sampler = torch_em.data.sampler.MinForegroundSampler(
        min_fraction=0.001,  # Require at least 1% of patch to be foreground
        background_id=0,     # Background label is 0
        p_reject=1.0         # Always reject patches below threshold
    )

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=label_transform,
        sampler=sampler,  # Add sampler to reject empty patches
        num_workers=8, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def run_training(checkpoint_name, model_type, train_instance_segmentation, nnunet_dataset_path,
                 n_epochs=20, batch_size=1, patch_size=512, n_objects_per_batch=25):
    """Run the actual model training.
    
    Args:
        checkpoint_name: Name of the checkpoint
        model_type: Type of SAM model (e.g., 'vit_b', 'vit_h')
        train_instance_segmentation: Whether to train instance segmentation
        nnunet_dataset_path: Path to the nnUNet dataset root directory
        n_epochs: Number of training epochs
        batch_size: Training batch size
        patch_size: Patch size for training
        n_objects_per_batch: Number of objects per batch to be sampled
    """

    # All hyperparameters for training.
    patch_shape = (1, patch_size, patch_size)  # the size of patches for training
    device = torch.device("cuda")  # the device used for training

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation, nnunet_dataset_path)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation, nnunet_dataset_path)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model.
    
    Args:
        checkpoint_name: Name of the checkpoint
        model_type: Type of SAM model
    """
    # export the model after training so that it can be used by the rest of the 'micro_sam' library
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    # Put the exported model in the checkpoint directory
    export_path = os.path.join("checkpoints", checkpoint_name, f"{checkpoint_name}_exported.pth")
    
    print(f"\nExporting model...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Export to: {export_path}")
    
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )
    
    print(f"Model exported successfully to {export_path}")


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from nnUNet datasets.
    It will automatically check for 2d_slices folder and create it if needed.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Segment Anything Model (SAM) with nnUNet dataset."
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the nnUNet dataset root directory (e.g., /path/to/Dataset005_MitoHardKidney)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h", "vit_t", "vit_b_lm", "vit_l_lm", "vit_h_lm"],
        help="The SAM model type to use. Default: vit_b. Note: vit_h usually yields higher quality but trains slower."
    )
    
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name for the checkpoint. If not provided, will be auto-generated from dataset name."
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of training epochs. Default: 20"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size. Default: 1"
    )
    
    parser.add_argument(
        "--patch_size",
        type=int,
        default=512,
        help="Patch size for training. Default: 512"
    )
    
    parser.add_argument(
        "--n_objects_per_batch",
        type=int,
        default=25,
        help="Number of objects per batch to be sampled. Default: 25"
    )
    
    parser.add_argument(
        "--no_instance_segmentation",
        action="store_true",
        help="Disable training additional convolutional decoder for end-to-end automatic instance segmentation"
    )
    
    parser.add_argument(
        "--skip_export",
        action="store_true",
        help="Skip exporting the model after training"
    )
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    
    # Auto-generate checkpoint name from dataset directory name if not provided
    if args.checkpoint_name is None:
        dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
        checkpoint_name = f"sam_{args.model_type}_{dataset_name}"
        print(f"Auto-generated checkpoint name: {checkpoint_name}")
    else:
        checkpoint_name = args.checkpoint_name

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = not args.no_instance_segmentation
    
    print("\n" + "="*60)
    print("SAM Fine-tuning Configuration:")
    print("="*60)
    print(f"  Dataset Path: {args.dataset_path}")
    print(f"  Model Type: {args.model_type}")
    print(f"  Checkpoint Name: {checkpoint_name}")
    print(f"  Number of Epochs: {args.n_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Patch Size: {args.patch_size}")
    print(f"  Objects per Batch: {args.n_objects_per_batch}")
    print(f"  Instance Segmentation: {train_instance_segmentation}")
    print("="*60 + "\n")
    
    # Run training
    run_training(
        checkpoint_name=checkpoint_name,
        model_type=args.model_type,
        train_instance_segmentation=train_instance_segmentation,
        nnunet_dataset_path=args.dataset_path,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        n_objects_per_batch=args.n_objects_per_batch
    )
    
    # Export model
    if not args.skip_export:
        export_model(checkpoint_name, args.model_type)
    else:
        print("\nSkipping model export as requested.")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Checkpoint saved to: ./checkpoints/{checkpoint_name}/")
    if not args.skip_export:
        print(f"Exported model: ./checkpoints/{checkpoint_name}/{checkpoint_name}_exported.pth")
    print("="*60)


if __name__ == "__main__":
    main()
