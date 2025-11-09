#!/usr/bin/env python
"""
Unified inference script for mitoem2.

Supports inference for all three methods: MitoNet, MicroSAM, and nnUNet.
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
import SimpleITK as sitk

from mitoem2.configs import load_config
from mitoem2.utils.logging import setup_logger
from mitoem2.utils.paths import get_data_root


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run mitoem2 inference")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["mitonet", "microsam", "nnunet"],
        help="Method to use for inference",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID or name (overrides config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--save-axis-results",
        action="store_true",
        help="Save individual axis predictions (xy, xz, yz) in addition to consensus",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config, method=args.method)
    else:
        # Use default config
        if args.method == "mitonet":
            config_path = Path(__file__).parent.parent / "configs" / "mitonet" / "inference_default.yaml"
        elif args.method == "microsam":
            config_path = Path(__file__).parent.parent / "configs" / "microsam" / "inference_default.yaml"
        else:
            raise ValueError(f"No default config for method: {args.method}")
        config = load_config(config_path, method=args.method)

    # Override with command-line arguments
    if args.dataset:
        config.dataset.id = args.dataset
    if args.checkpoint:
        config.model.checkpoint = args.checkpoint

    # Setup logging
    logger = setup_logger(
        name=f"mitoem2_inference_{args.method}",
        use_wandb=False,
    )

    logger.info(f"Starting inference with {args.method}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load input
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        # Single file inference
        image = _load_image(input_path)
        logger.info(f"Loaded image with shape: {image.shape}")

        # Run inference
        if args.method == "mitonet":
            from mitoem2.inference.mitonet_inference import MitoNetInferenceEngine
            from mitoem2.models.mitonet.model import MitoNetModel

            model = MitoNetModel(config_path=config.model.config_path)
            if config.model.checkpoint:
                model.load_weights(Path(config.model.checkpoint))

            engine = MitoNetInferenceEngine(
                model=model,
                config=config.inference.__dict__,
            )
            segmentation = engine.predict(image, save_axis_results=args.save_axis_results)

            # Save axis results if requested
            if args.save_axis_results and hasattr(engine, 'axis_results'):
                logger.info("Saving individual axis predictions...")
                file_ext = ''.join(input_path.suffixes) if input_path.suffixes else input_path.suffix
                if file_ext in [".nii.gz", ".nii"]:
                    base_name = input_path.name.replace(file_ext, "").replace("_0000", "")
                else:
                    base_name = input_path.stem
                for axis_name, axis_seg in engine.axis_results.items():
                    axis_output_file = output_path / f"{base_name}_{args.method}_{axis_name}_prediction{file_ext}"
                    _save_image(axis_seg, axis_output_file)
                    logger.info(f"Saved {axis_name} prediction to {axis_output_file}")

        elif args.method == "microsam":
            from mitoem2.inference.microsam_inference import MicroSAMInferenceEngine
            from mitoem2.models.microsam.model import MicroSAMModel

            model = MicroSAMModel(
                model_type=config.model.model_type,
                checkpoint_path=Path(config.model.checkpoint) if config.model.checkpoint else None,
            )
            engine = MicroSAMInferenceEngine(
                model=model,
                config=config.inference.__dict__,
            )
            segmentation = engine.predict(image)

        elif args.method == "nnunet":
            logger.info("nnUNet inference should be done using nnunetv2_predict command")
            logger.info(f"Example: nnunetv2_predict -i {input_path} -o {output_path} -d {config.dataset.id} -f 0")
            return

        # Save output - use method name in filename
        # Get file extension from input file
        file_ext = ''.join(input_path.suffixes) if input_path.suffixes else input_path.suffix
        # For .nii.gz files, preserve the full extension
        if file_ext in [".nii.gz", ".nii"]:
            base_name = input_path.name.replace(file_ext, "").replace("_0000", "")
            output_file = output_path / f"{base_name}_{args.method}_prediction{file_ext}"
            # For NIfTI files, preserve metadata if available
            if file_ext in [".nii.gz", ".nii"]:
                import SimpleITK as sitk
                # Try to load original image to get metadata
                try:
                    original_img = sitk.ReadImage(str(input_path))
                    sitk_seg = sitk.GetImageFromArray(segmentation.astype(np.uint16))
                    sitk_seg.SetSpacing(original_img.GetSpacing())
                    sitk_seg.SetOrigin(original_img.GetOrigin())
                    sitk_seg.SetDirection(original_img.GetDirection())
                    sitk.WriteImage(sitk_seg, str(output_file))
                    logger.info(f"Saved prediction to {output_file} (NIfTI format with metadata)")
                except Exception as e:
                    logger.warning(f"Could not preserve NIfTI metadata: {e}, saving without metadata")
                    _save_image(segmentation.astype(np.uint16), output_file)
            else:
                _save_image(segmentation.astype(np.uint16), output_file)
        else:
            # For TIFF and other formats
            output_file = output_path / f"{input_path.stem}_{args.method}_prediction{file_ext}"
            _save_image(segmentation.astype(np.uint16), output_file)
            logger.info(f"Saved prediction to {output_file}")

    else:
        # Directory inference
        logger.info("Directory inference not yet fully implemented")
        logger.info("Please process files individually or use method-specific tools")

    logger.info("Inference complete!")


def _load_image(path: Path) -> np.ndarray:
    """Load image from file."""
    path = Path(path)
    if path.suffix in [".nii", ".nii.gz"]:
        img_sitk = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img_sitk)
    else:
        return tiff.imread(str(path))


def _save_image(image: np.ndarray, path: Path) -> None:
    """Save image to file."""
    path = Path(path)
    print(f"Saving image to {path}, shape: {image.shape}")
    if path.suffix in [".nii", ".nii.gz"]:
        img_sitk = sitk.GetImageFromArray(image)
        sitk.WriteImage(img_sitk, str(path))
    else:
        tiff.imwrite(str(path), image, compression="zlib")


if __name__ == "__main__":
    main()
