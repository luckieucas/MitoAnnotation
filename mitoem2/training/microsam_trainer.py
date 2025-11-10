"""
MicroSAM trainer implementation.

Adapts the legacy `micro_sam_finetune.py` pipeline to the unified training interface.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader

from mitoem2.configs import MicroSAMConfig
from mitoem2.data.converters.nnunet_converter import convert_nnunet_to_empanada_format  # reuse util
from mitoem2.evaluation.evaluate_res import evaluate_single_file
from mitoem2.inference.microsam_inference import MicroSAMInferenceEngine
from mitoem2.models.microsam.model import MicroSAMModel
from mitoem2.training.trainer import BaseTrainer
from mitoem2.utils.logging import get_logger

logger = get_logger(__name__)


class MicroSAMTrainer(BaseTrainer):
    """
    MicroSAM trainer wrapping the micro_sam training utilities.

    1. Convert 3D nnUNet volumes to 2D slices (if necessary).
    2. Create torch_em dataloaders for train/validation.
    3. Call `micro_sam.training.train_sam`.
    4. (Optional) Export the model and run inference/evaluation on the test set.
    """

    def __init__(
        self,
        config: MicroSAMConfig,
        model: Optional[MicroSAMModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        use_microsam_training: bool = True,
    ):
        self.config = config
        self.use_microsam_training = use_microsam_training
        self.dataset_name = self._resolve_dataset_name()
        self.checkpoint_root = self._init_checkpoint_root()

        if model is None:
            model = MicroSAMModel(
                model_type=config.model.model_type,
                checkpoint_path=Path(config.model.checkpoint) if config.model.checkpoint else None,
            )

        super().__init__(
            model=model,
            train_loader=train_loader or DataLoader([]),
            val_loader=val_loader,
            optimizer=None,
            criterion=None,
        )

        self.sam_training = self._import_module("micro_sam.training")
        self.torch_em = self._import_module("torch_em")
        self.segmentation_dataset = self._import_from_module(
            "torch_em.data.segmentation_dataset", "SegmentationDataset"
        )
        self.MinForegroundSampler = self._import_from_module(
            "torch_em.data.sampler", "MinForegroundSampler"
        )
        self.PerObjectDistanceTransform = self._import_from_module(
            "torch_em.transform.label", "PerObjectDistanceTransform"
        )

    @staticmethod
    def _import_module(name: str):
        try:
            return __import__(name, fromlist=["*"])
        except ImportError as exc:
            raise ImportError(f"{name} is required for MicroSAM training.") from exc

    @staticmethod
    def _import_from_module(module_name: str, attr: str):
        module = MicroSAMTrainer._import_module(module_name)
        if not hasattr(module, attr):
            raise ImportError(f"{attr} not found in {module_name}")
        return getattr(module, attr)

    def _resolve_dataset_name(self) -> str:
        dataset_id = getattr(self.config.dataset, "id", None)
        if dataset_id is None:
            return "DatasetUnknown"
        try:
            nnunet_utils = self._import_module("nnunetv2.utilities.dataset_name_id_conversion")
            dataset_name = nnunet_utils.maybe_convert_to_dataset_name(dataset_id)
            if dataset_name:
                return str(dataset_name)
        except ImportError:
            logger.warning(
                "nnunetv2 is not available to resolve dataset name; falling back to dataset id."
            )
        return str(dataset_id)

    def _init_checkpoint_root(self) -> Path:
        base_dir = Path(self.config.output.model_dir) if self.config.output.model_dir else Path("checkpoints") / "microsam"
        checkpoint_root = base_dir / self.dataset_name
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.config.output.model_dir = str(checkpoint_root)
        return checkpoint_root

    @classmethod
    def from_config(cls, config: MicroSAMConfig) -> "MicroSAMTrainer":
        return cls(config=config)

    # ------------------------------------------------------------------ #
    # BaseTrainer interface (not used but kept for completeness)
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.use_microsam_training:
            raise NotImplementedError(
                "train_step is not used with MicroSAM training. Call `train()` to run the pipeline."
            )
        return {"loss": 0.0}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.use_microsam_training:
            raise NotImplementedError(
                "validate_step is not used with MicroSAM training. Call `train()` to run the pipeline."
            )
        return {"loss": 0.0}

    # ------------------------------------------------------------------ #
    # Pipeline helpers
    # ------------------------------------------------------------------ #
    def _resolve_nnunet_dataset_path(self) -> Path:
        dataset_id = self.config.dataset.id
        if dataset_id is None:
            raise ValueError("Dataset id must be specified for MicroSAM training.")

        try:
            nnunet_paths = self._import_module("nnunetv2.paths")
            nnunet_utils = self._import_module("nnunetv2.utilities.dataset_name_id_conversion")

            dataset_name = nnunet_utils.maybe_convert_to_dataset_name(dataset_id)
            dataset_path = Path(nnunet_paths.nnUNet_raw) / dataset_name
            if not dataset_path.exists():
                raise FileNotFoundError
            return dataset_path
        except (ImportError, FileNotFoundError):
            if self.config.dataset.root:
                dataset_path = Path(self.config.dataset.root)
                if dataset_path.exists():
                    return dataset_path
            raise ValueError(
                "Could not resolve nnUNet dataset path. Install nnUNetv2 or specify `dataset.root`."
            )

    def _ensure_2d_slices(self, dataset_path: Path) -> Path:
        """
        Convert nnUNet 3D data to 2D slices (if needed).

        The conversion mirrors `convert_3d_to_2d_slices` in the legacy micro_sam script.
        """
        slices_path = dataset_path / "2d_slices"
        skip_conversion = getattr(self.config, "skip_conversion", False)

        if skip_conversion and slices_path.exists():
            logger.info("Using existing 2D slices at %s", slices_path)
            return slices_path

        if slices_path.exists():
            logger.info("Existing 2D slices found at %s", slices_path)
            return slices_path

        logger.info("2D slices not found, converting nnUNet dataset at %s", dataset_path)
        convert_nnunet_to_empanada_format(
            nnunet_dataset_path=dataset_path, output_path=slices_path
        )
        return slices_path

    def _collect_volume_dirs(self, slices_path: Path, split: str) -> Tuple[List[Path], List[Path]]:
        split_dir = slices_path / split
        image_dirs = sorted(split_dir.glob("*/images"))
        mask_dirs = sorted(split_dir.glob("*/masks"))
        if not image_dirs:
            raise ValueError(f"No image directories found in {split_dir}")
        if not mask_dirs:
            raise ValueError(f"No mask directories found in {split_dir}")

        logger.info("Found %d volume(s) for %s split", len(image_dirs), split)
        return image_dirs, mask_dirs

    def _get_dataloader(
        self,
        split: str,
        patch_shape: Tuple[int, int, int],
        batch_size: int,
        train_instance_segmentation: bool,
        dataset_path: Path,
    ):
        slices_path = self._ensure_2d_slices(dataset_path)
        image_dirs, mask_dirs = self._collect_volume_dirs(slices_path, split)

        raw_paths = [str(path) for path in image_dirs]
        label_paths = [str(path) for path in mask_dirs]
        raw_key = label_key = "*.tif"

        if train_instance_segmentation:
            label_transform = self.PerObjectDistanceTransform(
                distances=True,
                boundary_distances=True,
                directed_distances=False,
                foreground=True,
                instances=True,
                min_size=25,
            )
        else:
            label_transform = self.torch_em.transform.label.connected_components

        sampler = self.MinForegroundSampler(
            min_fraction=0.05, background_id=0, p_reject=1.0
        )

        return self.torch_em.default_segmentation_loader(
            raw_paths=raw_paths,
            raw_key=raw_key,
            label_paths=label_paths,
            label_key=label_key,
            patch_shape=patch_shape,
            batch_size=batch_size,
            ndim=2,
            is_seg_dataset=True,
            label_transform=label_transform,
            sampler=sampler,
            num_workers=8,
            shuffle=True,
            raw_transform=self.sam_training.identity,
        )

    def _relocate_checkpoint_dir(self, checkpoint_name: str, target_dir: Path) -> None:
        """
        Ensure the checkpoint directory created by micro_sam training is moved
        into the unified checkpoints hierarchy.
        """
        candidate_dirs = [
            Path("checkpoints") / checkpoint_name,
            Path.cwd() / "checkpoints" / checkpoint_name,
        ]
        for candidate in candidate_dirs:
            if candidate.exists() and candidate != target_dir:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(candidate), str(target_dir))
                logger.info("Moved MicroSAM checkpoint directory %s -> %s", candidate, target_dir)
                break
        else:
            if not target_dir.exists():
                logger.warning(
                    "No checkpoint directory found for %s; expected one of %s",
                    checkpoint_name,
                    ", ".join(str(c) for c in candidate_dirs),
                )

    def _locate_best_checkpoint(self, checkpoint_name: str, target_dir: Path) -> Path:
        """
        Locate the `best.pt` checkpoint saved by micro_sam training.

        The micro_sam training code typically writes checkpoints under
        `./checkpoints/<name>/best.pt`. We search common locations to find it.
        """
        candidates = [
            target_dir / "best.pt",
            target_dir / "checkpoints" / "best.pt",
            Path("checkpoints") / checkpoint_name / "best.pt",
            Path.cwd() / "checkpoints" / checkpoint_name / "best.pt",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.info("Found micro_sam best checkpoint at %s", candidate)
                return candidate

        raise FileNotFoundError(
            "Could not locate micro_sam checkpoint 'best.pt'. Checked:\n"
            + "\n".join(str(c) for c in candidates)
        )

    def _compute_training_params(self) -> Dict[str, Any]:
        training_cfg = self.config.training
        params = {
            "n_epochs": getattr(training_cfg, "n_epochs", None) or getattr(training_cfg, "num_epochs", None) or 20,
            "batch_size": getattr(training_cfg, "batch_size", None) or 1,
            "patch_size": getattr(training_cfg, "patch_size", None) or 512,
            "n_objects_per_batch": getattr(training_cfg, "n_objects_per_batch", None) or 25,
            "train_instance_segmentation": getattr(training_cfg, "train_instance_segmentation", None),
        }
        if params["train_instance_segmentation"] is None:
            params["train_instance_segmentation"] = True
        return params

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(
        self,
        num_epochs: Optional[int] = None,
        start_epoch: int = 0,  # unused but kept for compatibility
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        if not self.use_microsam_training:
            return super().train(
                num_epochs=num_epochs or 1,
                start_epoch=start_epoch,
                checkpoint_path=checkpoint_path,
            )

        dataset_path = self._resolve_nnunet_dataset_path()
        train_params = self._compute_training_params()
        if num_epochs is not None:
            train_params["n_epochs"] = num_epochs

        model_type = (
            self.config.model.model_type
            if hasattr(self.config.model, "model_type") and self.config.model.model_type
            else "vit_b"
        )

        base_dir = Path(self.config.output.model_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"sam_{model_type}_{dataset_path.name}"
        target_checkpoint_dir = base_dir / checkpoint_name
        target_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("MicroSAM Fine-tuning Configuration")
        logger.info("=" * 60)
        logger.info("  Dataset Path: %s", dataset_path)
        logger.info("  Model Type: %s", model_type)
        logger.info("  Checkpoint Name: %s", checkpoint_name)
        logger.info("  Number of Epochs: %s", train_params['n_epochs'])
        logger.info("  Batch Size: %s", train_params['batch_size'])
        logger.info("  Patch Size: %s", train_params['patch_size'])
        logger.info("  Objects per Batch: %s", train_params['n_objects_per_batch'])
        logger.info("  Instance Segmentation: %s", train_params['train_instance_segmentation'])
        logger.info("=" * 60)

        patch_shape = (1, train_params["patch_size"], train_params["patch_size"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Creating dataloaders ...")
        train_loader = self._get_dataloader(
            "train",
            patch_shape,
            train_params["batch_size"],
            train_params["train_instance_segmentation"],
            dataset_path,
        )
        val_loader = self._get_dataloader(
            "val",
            patch_shape,
            train_params["batch_size"],
            train_params["train_instance_segmentation"],
            dataset_path,
        )

        logger.info("Starting micro_sam training ...")
        # self.sam_training.train_sam(
        #     name=checkpoint_name,
        #     model_type=model_type,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     n_epochs=train_params["n_epochs"],
        #     n_objects_per_batch=train_params["n_objects_per_batch"],
        #     with_segmentation_decoder=train_params["train_instance_segmentation"],
        #     device=device,
        # )
        logger.info("MicroSAM training finished.")

        self._relocate_checkpoint_dir(checkpoint_name, target_checkpoint_dir)
        best_checkpoint = self._locate_best_checkpoint(checkpoint_name, target_checkpoint_dir)
        if getattr(self.config, "test_after_training", False):
            logger.info("Running test after training ...")
            self._test_after_training(
                best_checkpoint=best_checkpoint,
                dataset_path=dataset_path,
                checkpoint_name=checkpoint_name,
                model_type=model_type,
                output_dir=target_checkpoint_dir,
            )

    # ------------------------------------------------------------------ #
    # Evaluation and inference utilities
    # ------------------------------------------------------------------ #
    def _collect_test_images(self, dataset_path: Path) -> List[Path]:
        test_images_dir = dataset_path / "imagesTs"
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

        image_files: List[str] = []
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.nii", "*.nii.gz"]:
            image_files.extend(glob(str(test_images_dir / ext)))
        if not image_files:
            raise FileNotFoundError(f"No test images found in {test_images_dir}")
        return [Path(f) for f in image_files]

    def _load_image_with_metadata(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        file_ext = "".join(path.suffixes)
        metadata: Dict[str, Any] = {"ext": file_ext}
        if file_ext in [".nii.gz", ".nii"]:
            import SimpleITK as sitk

            sitk_image = sitk.ReadImage(str(path))
            image = sitk.GetArrayFromImage(sitk_image)
            metadata["sitk"] = sitk_image
            metadata["spacing"] = sitk_image.GetSpacing()
            metadata["origin"] = sitk_image.GetOrigin()
            metadata["direction"] = sitk_image.GetDirection()
        else:
            image = tiff.imread(str(path))
        return image, metadata

    def _save_prediction(self, prediction: np.ndarray, input_path: Path, output_dir: Path, metadata: Dict[str, Any]) -> Path:
        file_ext = metadata.get("ext", input_path.suffix)
        if file_ext == ".nii":
            base_name = input_path.stem.replace("_0000", "")
        elif file_ext == ".nii.gz":
            base_name = input_path.name.replace(".nii.gz", "").replace("_0000", "")
        else:
            base_name = input_path.stem.replace("_0000", "")

        output_file = output_dir / f"{base_name}{file_ext}"

        if file_ext in [".nii.gz", ".nii"]:
            import SimpleITK as sitk

            sitk_image = metadata.get("sitk")
            if sitk_image is None:
                sitk_image = sitk.GetImageFromArray(prediction.astype(np.uint16))
            else:
                sitk_image = sitk.GetImageFromArray(prediction.astype(np.uint16))
                sitk_image.SetSpacing(metadata.get("spacing", sitk_image.GetSpacing()))
                sitk_image.SetOrigin(metadata.get("origin", sitk_image.GetOrigin()))
                sitk_image.SetDirection(metadata.get("direction", sitk_image.GetDirection()))
            sitk.WriteImage(sitk_image, str(output_file))
        else:
            data_to_save = prediction.astype(np.uint16)
            if data_to_save.ndim == 3:
                data_to_save = data_to_save.transpose(1, 0, 2)
            tiff.imwrite(str(output_file), data_to_save, compression="zlib")
        return output_file

    def _test_after_training(
        self,
        best_checkpoint: Path,
        dataset_path: Path,
        checkpoint_name: str,
        model_type: str,
        output_dir: Path,
    ) -> None:
        model = MicroSAMModel(model_type=model_type, checkpoint_path=best_checkpoint)
        inference_config: Dict[str, Any] = {}
        if hasattr(self.config, "inference") and self.config.inference:
            inference_config = asdict(self.config.inference)

        train_inst = self._compute_training_params()["train_instance_segmentation"]
        if not train_inst:
            inference_config["use_amg"] = True

        engine = MicroSAMInferenceEngine(model=model, config=inference_config)

        test_images = self._collect_test_images(dataset_path)

        mode = (getattr(self.config, "mode", None) or "FT").upper()
        predictions_dir = dataset_path / f"imagesTs_microsam_{mode}_pred"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving predictions to %s", predictions_dir)

        use_embeddings = bool(inference_config.get("use_embeddings", False))
        label_path_cfg = inference_config.get("label_path")
        label_mask = None
        if label_path_cfg:
            label_path = Path(label_path_cfg)
            if label_path.exists():
                try:
                    label_mask, _ = self._load_image_with_metadata(label_path)
                    logger.info("Loaded label mask for filtering from %s", label_path)
                except Exception as exc:
                    logger.warning("Failed to load label mask from %s: %s", label_path, exc)
                    label_mask = None
            else:
                logger.warning("Label path specified but not found: %s", label_path)

        generated_files: List[Path] = []

        for image_path in test_images:
            try:
                logger.info("Running inference for %s", image_path.name)
                image, metadata = self._load_image_with_metadata(image_path)

                base_name = metadata.get("base_name")
                if base_name is None:
                    if metadata.get("ext") == ".nii.gz":
                        base_name = image_path.name.replace(".nii.gz", "").replace("_0000", "")
                    elif metadata.get("ext") == ".nii":
                        base_name = image_path.stem.replace("_0000", "")
                    else:
                        base_name = image_path.stem.replace("_0000", "")
                    metadata["base_name"] = base_name

                embedding_path = None
                if use_embeddings:
                    embedding_path = predictions_dir / f"{base_name}_embeddings.zarr"

                prediction = engine.predict(
                    image,
                    embedding_path=str(embedding_path) if embedding_path else None,
                )

                if label_mask is not None:
                    if label_mask.shape == prediction.shape:
                        prediction[label_mask == 0] = 0
                    else:
                        logger.warning(
                            "Label mask shape %s does not match prediction shape %s for %s",
                            label_mask.shape,
                            prediction.shape,
                            image_path.name,
                        )

                pred_file = self._save_prediction(prediction, image_path, predictions_dir, metadata)
                generated_files.append(pred_file)
            except Exception as exc:
                logger.error("Error processing %s: %s", image_path.name, exc)

        if not generated_files:
            logger.warning("No predictions were generated, skipping evaluation.")
            return

        gt_dir = dataset_path / "instancesTs"
        if not gt_dir.exists():
            gt_dir = dataset_path / "labelsTs"
        mask_dir = dataset_path / "masksTs"
        mask_dir_path = mask_dir if mask_dir.exists() else None

        if gt_dir.exists() and getattr(self.config, "eval", True):
            self._evaluate_predictions(predictions_dir, generated_files, gt_dir, mask_dir_path)
        else:
            logger.info("Skipping evaluation (ground truth missing or eval disabled).")

    def _evaluate_predictions(
        self,
        predictions_dir: Path,
        prediction_files: Iterable[Path],
        gt_dir: Path,
        mask_dir: Optional[Path],
    ) -> None:
        results: List[Dict[str, Any]] = []

        for pred_file in prediction_files:
            base_name = pred_file.name.replace("_prediction", "")
            candidates = [
                gt_dir / base_name,
                gt_dir / base_name.replace(".nii.gz", ".nii"),
                gt_dir / base_name.replace(".nii", ".nii.gz"),
                gt_dir / base_name.replace(".tiff", ".tif"),
                gt_dir / base_name.replace(".tif", ".tiff"),
            ]
            gt_file = next((c for c in candidates if c.exists()), None)

            if gt_file is None:
                logger.warning("No matching ground truth found for %s", pred_file.name)
                continue

            mask_file = None
            if mask_dir is not None:
                mask_candidates = [
                    mask_dir / base_name,
                    mask_dir / base_name.replace(".nii.gz", ".nii"),
                    mask_dir / base_name.replace(".nii", ".nii.gz"),
                    mask_dir / base_name.replace(".tiff", ".tif"),
                    mask_dir / base_name.replace(".tif", ".tiff"),
                ]
                mask_file = next((c for c in mask_candidates if c.exists()), None)

            try:
                eval_result = evaluate_single_file(
                    pred_file=str(pred_file),
                    gt_file=str(gt_file),
                    mask_file=str(mask_file) if mask_file else None,
                    save_results=True,
                )
                results.append(
                    {
                        "prediction": str(pred_file),
                        "ground_truth": str(gt_file),
                        "metrics": eval_result,
                    }
                )
                logger.info(
                    "Evaluated %s | Accuracy=%.4f Binary Accuracy=%.4f F1=%.4f",
                    pred_file.name,
                    eval_result.get("accuracy", 0.0),
                    eval_result.get("binary_accuracy", 0.0),
                    eval_result.get("f1", 0.0),
                )
            except Exception as exc:
                logger.error("Failed to evaluate %s: %s", pred_file, exc)

        if results:
            summary_path = predictions_dir / "evaluation_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self._convert_numpy_types(results), f, indent=2)
            logger.info("Saved evaluation summary to %s", summary_path)

    @staticmethod
    def _convert_numpy_types(obj: Any) -> Any:
        import numpy as np

        if isinstance(obj, dict):
            return {k: MicroSAMTrainer._convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MicroSAMTrainer._convert_numpy_types(v) for v in obj]
        if isinstance(obj, tuple):
            return [MicroSAMTrainer._convert_numpy_types(v) for v in obj]
        if isinstance(obj, (np.integer, np.unsignedinteger)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


__all__ = ["MicroSAMTrainer"]

