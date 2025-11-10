"""
nnUNet trainer wrapper.

Migrates the legacy `auto_train_nnunet` pipeline into the unified training interface.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
import shutil  # Keep import

import tifffile as tiff
import numpy as np

from mitoem2.configs import NNUNetConfig
from mitoem2.utils.logging import get_logger
from mitoem2.evaluation.evaluate_res import evaluate_single_file
from mitoem2.postprocessing.bc_watershed import process_folder


class NNUNetTrainer:
    """
    nnUNet training pipeline implemented directly in the mitoem2 package.
    """

    def __init__(self, config: NNUNetConfig, logger=None):
        if not isinstance(config, NNUNetConfig):
            raise TypeError(f"Expected NNUNetConfig, got {type(config)}")

        self.config = config
        self.logger = logger or get_logger(__name__)
        self.pipeline_config = config.nnunet

        self.dataset_id = self._require_dataset_id()
        self.dataset_name = self._resolve_dataset_name(self.dataset_id)

        self._init_paths()

    @classmethod
    def from_config(cls, config: NNUNetConfig, logger=None) -> "NNUNetTrainer":
        """Create trainer instance from configuration."""
        return cls(config=config, logger=logger)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_pipeline(self) -> None:
        """Run the nnUNet training pipeline following configuration."""
        start_time = time.time()
        self.logger.info("Starting nnUNet pipeline")
        self.logger.info("  Dataset ID: %s", self.dataset_id)
        self.logger.info("  nnUNet raw dir: %s", self.nnunet_raw_dir)
        self.logger.info("  nnUNet preprocessed dir: %s", self.nnunet_preprocessed_dir)
        self.logger.info("  nnUNet results dir: %s", self.nnunet_results_dir)

        try:
            if not self.pipeline_config.skip_boundary:
                self.logger.info("=== Step 1: Generate boundary masks ===")
                self._generate_boundary_masks()
            else:
                self.logger.info("=== Step 1: Skip boundary mask generation ===")

            if not self.pipeline_config.skip_plan:
                self.logger.info("=== Step 2: nnUNet plan & preprocess ===")
                
                # --- Keep swap logic (start) ---
                self.logger.info("Swapping label directories for nnU-Net preprocessing...")
                self._swap_labels_for_preprocessing()
                
                try:
                    self._nnunet_plan_and_process()
                finally:
                    # Ensure directories are restored even if preprocessing fails
                    self.logger.info("Restoring original label directories...")
                    self._restore_labels_after_preprocessing()
                # --- Keep swap logic (end) ---
                    
            else:
                self.logger.info("=== Step 2: Skip nnUNet plan & preprocess ===")

            if not self.pipeline_config.skip_training:
                self.logger.info("=== Step 3: nnUNet training ===")
                self._nnunet_train()
            else:
                self.logger.info("=== Step 3: Skip nnUNet training ===")

            if not self.pipeline_config.skip_prediction:
                self.logger.info("=== Step 4: nnUNet prediction ===")
                self._nnunet_predict()
            else:
                self.logger.info("=== Step 4: Skip nnUNet prediction ===")

            if not self.pipeline_config.skip_postprocess:
                self.logger.info("=== Step 5: Post-process predictions ===")
                self._postprocess_predictions()
            else:
                self.logger.info("=== Step 5: Skip post-processing ===")
                self.pipeline_config.skip_evaluation = True

            if not self.pipeline_config.skip_evaluation:
                self.logger.info("=== Step 6: Evaluate predictions ===")
                self._evaluate_results()
            else:
                self.logger.info("=== Step 6: Skip evaluation ===")

        except Exception as exc:
            self.logger.error("nnUNet pipeline failed: %s", exc)
            raise
        finally:
            elapsed = time.time() - start_time
            self.logger.info("nnUNet pipeline finished in %.2f seconds", elapsed)

    # ------------------------------------------------------------------ #
    # Internal setup helpers
    # ------------------------------------------------------------------ #

    def _require_dataset_id(self) -> str:
        dataset_id = (
            self.config.dataset.id
            if hasattr(self.config.dataset, "id")
            else self.config.dataset.get("id")
        )
        if dataset_id is None:
            raise ValueError("Dataset ID is required for nnUNet pipeline.")
        return str(dataset_id)

    def _resolve_dataset_name(self, dataset_id: str) -> str:
        try:
            from nnunetv2.utilities.dataset_name_id_conversion import (
                maybe_convert_to_dataset_name,
            )
        except ImportError as exc:
            raise ImportError(
                "nnunetv2 utilities not available. Please ensure nnUNet v2 is installed."
            ) from exc

        dataset_name = maybe_convert_to_dataset_name(dataset_id)
        if dataset_name is None:
            raise ValueError(f"Could not resolve dataset name for id: {dataset_id}")
        return dataset_name

    def _init_paths(self) -> None:
        """
        Initialise nnUNet directory structure similar to the legacy pipeline.
        """
        try:
            from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
        except ImportError as exc:
            raise ImportError(
                "nnunetv2 must be installed and nnUNet paths configured. "
                "Please ensure nnUNet is available in the environment."
            ) from exc

        self.nnunet_raw_dir = Path(nnUNet_raw).expanduser().resolve()
        self.nnunet_preprocessed_dir = Path(nnUNet_preprocessed).expanduser().resolve()
        results_base = (
            Path(self.config.output.model_dir).expanduser().resolve()
            if self.config.output.model_dir
            else (Path("checkpoints") / "nnunet").resolve()
        )
        self.nnunet_results_dir = results_base
        self.nnunet_results_dir.mkdir(parents=True, exist_ok=True)
        self.config.output.model_dir = str(self.nnunet_results_dir)

        if not self.nnunet_raw_dir.exists():
            raise FileNotFoundError(f"nnUNet raw directory not found: {self.nnunet_raw_dir}")
        if not self.nnunet_preprocessed_dir.exists():
            raise FileNotFoundError(
                f"nnUNet preprocessed directory not found: {self.nnunet_preprocessed_dir}"
            )

        self.dataset_dir = self.nnunet_raw_dir / self.dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.images_tr_dir = self.dataset_dir / "imagesTr"
        
        # --- Keep path modifications (start) ---
        # self.labels_tr_dir is the generated boundary labels (target)
        self.labels_tr_dir = self.dataset_dir / "bc_labelsTr"
        # self.instances_tr_dir is the 'labelsTr' folder expected by nnU-Net (contains instances)
        self.instances_tr_dir = self.dataset_dir / "labelsTr"
        # self.instances_tr_dir_backup is the temporary backup path for 'labelsTr' folder
        self.instances_tr_dir_backup = self.dataset_dir / "labelsTr_TEMP_BACKUP"
        # --- Keep path modifications (end) ---

        self.images_ts_dir = self.dataset_dir / "imagesTs"
        self.instances_ts_dir = self.dataset_dir / "labelsTs"
        self.masks_ts_dir = self.dataset_dir / "masksTs"
        self.predictions_dir = self.dataset_dir / "imagesTs_pred"
        self.final_results_dir = self.dataset_dir / "imagesTs_pred_waterz"

        required_dirs = [
            self.images_tr_dir,
            self.instances_tr_dir,  # Check if 'labelsTr' (instances) exists
            self.images_ts_dir,
        ]
        for path in required_dirs:
            if not path.exists():
                raise FileNotFoundError(f"Required nnUNet directory not found: {path}")

        for path in [
            self.labels_tr_dir,  # Ensure 'bc_labelsTr' folder exists
            self.instances_ts_dir,
            self.masks_ts_dir,
            self.predictions_dir,
            self.final_results_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # --- Restore _generate_boundary_masks ---
    def _generate_boundary_masks(self) -> None:
        """
        Generate boundary masks for training data.
        (Restored to original 0, 1, 2 logic)
        Read from: self.instances_tr_dir (i.e., 'labelsTr', contains instances)
        Write to: self.labels_tr_dir (i.e., 'bc_labelsTr', used to store boundary labels)
        """
        try:
            from connectomics.data.utils.data_segmentation import seg_to_instance_bd
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError(
                "connectomics is required for boundary mask generation."
            ) from exc

        self.logger.info(
            "Generating boundary (0/1/2) masks from %s -> %s",
            self.instances_tr_dir,
            self.labels_tr_dir
        )
        # Ensure target folder (bc_labelsTr) exists
        self.labels_tr_dir.mkdir(parents=True, exist_ok=True)

        # Search for .nii.gz files
        label_files = sorted(self.instances_tr_dir.glob("*.nii.gz"))
        if not label_files:
            # If .nii.gz not found, try .nii
            label_files = sorted(self.instances_tr_dir.glob("*.nii"))
        
        for label_file in label_files:
            try:
                # Read NIfTI file using SimpleITK
                sitk_image = sitk.ReadImage(str(label_file))
                label_volume = sitk.GetArrayFromImage(sitk_image).astype(np.uint16)
                
                # Preserve original metadata
                spacing = sitk_image.GetSpacing()
                origin = sitk_image.GetOrigin()
                direction = sitk_image.GetDirection()
                
                # --- Original logic ---
                binary = (label_volume > 0).astype(np.uint8)
                contour = seg_to_instance_bd(binary, tsz_h=3)  # Assuming tsz_h=3 is desired
                contour[contour > 0] = 2

                combined = binary + contour
                combined[combined > 2] = 1  # Interior(1) + boundary(2) = 3, revert to interior(1)
                # --- End of logic ---

                # Save as .nii.gz using SimpleITK, preserving original metadata
                output_path = self.labels_tr_dir / label_file.name
                output_sitk = sitk.GetImageFromArray(combined.astype(np.uint8))
                output_sitk.SetSpacing(spacing)
                output_sitk.SetOrigin(origin)
                output_sitk.SetDirection(direction)
                sitk.WriteImage(output_sitk, str(output_path), useCompression=True)
                self.logger.debug("Updated boundary mask: %s", output_path.name)
            except Exception as e:
                self.logger.error(f"Failed to create boundary mask for {label_file.name}: {e}")

    # --- Keep _swap_labels_for_preprocessing ---
    def _swap_labels_for_preprocessing(self) -> None:
        """
        Swap label directories for preprocessing.
        1. Move {instances_tr_dir} (labelsTr) -> {instances_tr_dir_backup} (labelsTr_TEMP_BACKUP)
        2. Move {labels_tr_dir} (bc_labelsTr) -> {instances_tr_dir} (labelsTr)
        """
        if self.instances_tr_dir_backup.exists():
            self.logger.warning(
                "Backup directory %s already exists. "
                "This might be from a failed previous run. Attempting to clean up.",
                self.instances_tr_dir_backup,
            )
            shutil.rmtree(self.instances_tr_dir_backup)
        
        if not self.instances_tr_dir.exists():
                raise FileNotFoundError(f"Original instance directory {self.instances_tr_dir} not found for swap.")
        if not self.labels_tr_dir.exists():
            raise FileNotFoundError(f"Generated boundary directory {self.labels_tr_dir} not found for swap.")

        # 1. mv labelsTr -> labelsTr_TEMP_BACKUP
        shutil.move(str(self.instances_tr_dir), str(self.instances_tr_dir_backup))
        
        # 2. mv bc_labelsTr -> labelsTr
        shutil.move(str(self.labels_tr_dir), str(self.instances_tr_dir))
        self.logger.info("Label directories swapped for preprocessing.")

    # --- Keep _restore_labels_after_preprocessing ---
    def _restore_labels_after_preprocessing(self) -> None:
        """
        Restore original label directories.
        1. Move {instances_tr_dir} (labelsTr, contains boundaries) -> {labels_tr_dir} (bc_labelsTr)
        2. Move {instances_tr_dir_backup} (labelsTr_TEMP_BACKUP) -> {instances_tr_dir} (labelsTr)
        """
        if not self.instances_tr_dir.exists():
            self.logger.error(
                "Cannot restore: %s (boundary labels) is missing.", self.instances_tr_dir
            )
            return
        if not self.instances_tr_dir_backup.exists():
                self.logger.error(
                "Cannot restore: %s (instance backup) is missing.", self.instances_tr_dir_backup
            )
                return

        # 1. mv labelsTr -> bc_labelsTr
        shutil.move(str(self.instances_tr_dir), str(self.labels_tr_dir))
        
        # 2. mv labelsTr_TEMP_BACKUP -> labelsTr
        shutil.move(str(self.instances_tr_dir_backup), str(self.instances_tr_dir))
        self.logger.info("Original label directories restored.")


    def _nnunet_plan_and_process(self) -> None:
        env = self._nnunet_env()
        dataset_number = self._dataset_number()
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            dataset_number,
            "--verify_dataset_integrity",
            "-c",
            "3d_fullres",
        ]
        self._run_subprocess(cmd, env=env, description="nnUNet plan and preprocess")

    def _nnunet_train(self) -> None:
        env = self._nnunet_env()
        dataset_number = self._dataset_number()
        cmd = [
            "nnUNetv2_train",
            dataset_number,
            "3d_fullres",
            str(self.pipeline_config.fold),
        ]
        self._run_subprocess(cmd, env=env, description="nnUNet training")

    def _nnunet_predict(self) -> None:
        env = self._nnunet_env()
        dataset_number = self._dataset_number()
        cmd = [
            "nnUNetv2_predict",
            "-i",
            str(self.images_ts_dir),
            "-o",
            str(self.predictions_dir),
            "-d",
            dataset_number,
            "-c",
            "3d_fullres",
            "-f",
            str(self.pipeline_config.fold),
            "--save_probabilities",
        ]
        self.logger.info("Running prediction command: %s", " ".join(cmd))  # Add logging
        self._run_subprocess(cmd, env=env, description="nnUNet prediction")

    def _postprocess_predictions(self) -> None:
        self.logger.info(
            "Running postprocessing from %s to %s",
            self.predictions_dir,
            self.final_results_dir
        )
        process_folder(self.predictions_dir, self.final_results_dir, save_tiff=False, save_nii=True)

    def _evaluate_results(self) -> None:
        """
        Evaluate predictions produced by the nnUNet pipeline.
        """
        pred_dir = self.final_results_dir if any(self.final_results_dir.glob("*")) else self.predictions_dir
        self.logger.info("Evaluating results in: %s", pred_dir)  # Add logging
        if not pred_dir.exists():
            self.logger.warning("Prediction directory does not exist: %s", pred_dir)
            return

        prediction_files = []
        for pattern in ("*.tif", "*.tiff", "*.nii.gz", "*.nii"):
            prediction_files.extend(sorted(pred_dir.glob(pattern)))

        if not prediction_files:
            self.logger.warning("No prediction files found in %s", pred_dir)
            return

        eval_dir = pred_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)

        all_results = []

        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.unsignedinteger)):
                return int(obj)
            if isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            if hasattr(obj, "item"):
                try:
                    return convert_numpy_types(obj.item())
                except (ValueError, AttributeError):
                    return str(obj)
            return obj

        for pred_file in prediction_files:
            file_ext = "".join(pred_file.suffixes) if pred_file.suffixes else pred_file.suffix
            if file_ext == ".nii.gz":
                # Fix: Ensure .nii.gz is correctly replaced
                base_name = pred_file.name.replace("_seg", "").replace("_prediction", "").replace(".nii.gz", "")
            else:
                base_name = pred_file.stem.replace("_seg", "").replace("_prediction", "")

            gt_file = None
            mask_file = None

            # Fix: Ensure candidate_exts includes original suffix
            candidate_exts = [".tiff", ".tif", ".nii.gz", ".nii"]
            if file_ext not in candidate_exts:
                candidate_exts.insert(0, file_ext)
            
            # Remove duplicates
            candidate_exts = sorted(list(set(candidate_exts)))


            for ext in candidate_exts:
                candidate_gt = self.instances_ts_dir / f"{base_name}{ext}"
                if candidate_gt.exists():
                    gt_file = candidate_gt
                    break

            if gt_file is None:
                self.logger.warning("No ground truth found for %s (Base name: %s, Searched in: %s)", 
                                    pred_file.name, base_name, self.instances_ts_dir)
                continue

            for ext in candidate_exts:
                candidate_mask = self.masks_ts_dir / f"{base_name}{ext}"
                if candidate_mask.exists():
                    mask_file = candidate_mask
                    break
            
            # if mask_file:
            #     self.logger.debug("Found mask for %s: %s", pred_file.name, mask_file.name)

            self.logger.info("Evaluating %s against %s", pred_file.name, gt_file.name)
            try:
                results = evaluate_single_file(
                    pred_file=str(pred_file),
                    gt_file=str(gt_file),
                    mask_file=str(mask_file) if mask_file else None,
                    save_results=True,
                )
            except Exception as exc:
                self.logger.error("Error evaluating %s: %s", pred_file.name, exc)
                continue

            all_results.append(
                {
                    "pred_file": str(pred_file),
                    "gt_file": str(gt_file),
                    "mask_file": str(mask_file) if mask_file else None,
                    "results": results,
                }
            )

            self.logger.info(
                "Metrics for %s - PQ: %.4f, SQ: %.4f, RQ: %.4f, F1: %.4f",
                pred_file.name,
                results.get("PQ", float("nan")),
                results.get("SQ", float("nan")),
                results.get("RQ", float("nan")),
                results.get("f1", float("nan")),
            )

        if not all_results:
            self.logger.warning("No evaluation results were produced.")
            return

        serializable_results = convert_numpy_types(all_results)
        summary_file = eval_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        self.logger.info("Saved evaluation summary to %s", summary_file)

        metrics_to_avg = ["PQ", "SQ", "RQ", "IoU", "F1", "accuracy", "precision", "recall"]
        averaged = {}
        for metric in metrics_to_avg:
            values = []
            for result in all_results:
                metric_val = result["results"].get(metric)
                if metric_val is None:
                    continue
                if isinstance(metric_val, (np.integer, np.unsignedinteger)):
                    metric_val = int(metric_val)
                elif isinstance(metric_val, (np.floating,)):
                    metric_val = float(metric_val)
                values.append(metric_val)
            if values:
                averaged[f"avg_{metric}"] = sum(values) / len(values)

        if averaged:
            averaged_serializable = convert_numpy_types(averaged)
            averages_file = eval_dir / "evaluation_averages.json"
            with open(averages_file, "w") as f:
                json.dump(averaged_serializable, f, indent=2)
            self.logger.info("Saved averaged metrics to %s", averages_file)
            for metric, value in averaged.items():
                self.logger.info("%s: %.4f", metric, value)

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def _dataset_number(self) -> str:
        return str(int(self.dataset_id.split("_")[0].replace("Dataset", "")))

    def _nnunet_env(self) -> dict:
        env = os.environ.copy()
        env["nnUNet_raw"] = str(self.nnunet_raw_dir)
        env["nnUNet_preprocessed"] = str(self.nnunet_preprocessed_dir)
        env["RESULTS_FOLDER"] = str(self.nnunet_results_dir)
        return env

    def _run_subprocess(self, cmd, env=None, description: str = "command") -> None:
        self.logger.info("Running %s: %s", description, " ".join(map(str, cmd)))
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            self.logger.error("%s failed: %s", description, exc)
            raise


__all__ = ["NNUNetTrainer"]