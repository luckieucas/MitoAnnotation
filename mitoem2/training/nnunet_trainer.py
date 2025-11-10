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
                self._nnunet_plan_and_process()
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
        self.labels_tr_dir = self.dataset_dir / "labelsTr"
        self.instances_tr_dir = self.dataset_dir / "instancesTr"
        self.images_ts_dir = self.dataset_dir / "imagesTs"
        self.instances_ts_dir = self.dataset_dir / "instancesTs"
        self.masks_ts_dir = self.dataset_dir / "masksTs"
        self.predictions_dir = self.dataset_dir / "imagesTs_pred"
        self.final_results_dir = self.dataset_dir / "imagesTs_pred_waterz"
        self.boundary_masks_dir = self.dataset_dir / "boundary_masks"

        required_dirs = [
            self.images_tr_dir,
            self.labels_tr_dir,
            self.instances_tr_dir,
            self.images_ts_dir,
        ]
        for path in required_dirs:
            if not path.exists():
                raise FileNotFoundError(f"Required nnUNet directory not found: {path}")

        for path in [
            self.instances_ts_dir,
            self.masks_ts_dir,
            self.predictions_dir,
            self.final_results_dir,
            self.boundary_masks_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _generate_boundary_masks(self) -> None:
        """
        Generate boundary masks for training data.
        """
        try:
            from connectomics.data.utils.data_segmentation import seg_to_instance_bd
        except ImportError as exc:
            raise ImportError(
                "connectomics is required for boundary mask generation."
            ) from exc

        for label_file in sorted(self.instances_tr_dir.glob("*.tiff")):
            label_volume = tiff.imread(label_file).astype(np.uint16)
            binary = (label_volume > 0).astype(np.uint8)
            contour = seg_to_instance_bd(binary, tsz_h=3)
            contour[contour > 0] = 2

            combined = binary + contour
            combined[combined > 2] = 1

            output_path = self.labels_tr_dir / label_file.name
            tiff.imwrite(output_path, combined.astype(np.uint8), compression="zlib")
            self.logger.debug("Updated boundary mask: %s", output_path.name)

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
        self._run_subprocess(cmd, env=env, description="nnUNet prediction")

    def _postprocess_predictions(self) -> None:
        process_folder(self.predictions_dir, self.final_results_dir, save_tiff=False, save_nii=True)

    def _evaluate_results(self) -> None:
        """
        Evaluate predictions produced by the nnUNet pipeline.
        """
        pred_dir = self.final_results_dir if any(self.final_results_dir.glob("*")) else self.predictions_dir
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
                base_name = pred_file.name.replace("_seg", "").replace("_prediction", "").replace(file_ext, "")
            else:
                base_name = pred_file.stem.replace("_seg", "").replace("_prediction", "")

            gt_file = None
            mask_file = None

            candidate_exts = [file_ext] if file_ext else []
            if file_ext == ".nii.gz":
                candidate_exts.append(".nii")
            elif file_ext == ".nii":
                candidate_exts.append(".nii.gz")
            elif file_ext in {".tiff", ".tif"}:
                candidate_exts.extend([".tiff", ".tif"])
            else:
                candidate_exts.extend([".tiff", ".tif", ".nii.gz", ".nii"])

            for ext in candidate_exts:
                candidate_gt = self.instances_ts_dir / f"{base_name}{ext}"
                if candidate_gt.exists():
                    gt_file = candidate_gt
                    break

            if gt_file is None:
                self.logger.warning("No ground truth found for %s", pred_file.name)
                continue

            for ext in candidate_exts:
                candidate_mask = self.masks_ts_dir / f"{base_name}{ext}"
                if candidate_mask.exists():
                    mask_file = candidate_mask
                    break

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
                "Metrics for %s - accuracy: %.4f, precision: %.4f, recall: %.4f, F1: %.4f",
                pred_file.name,
                results.get("accuracy", float("nan")),
                results.get("precision", float("nan")),
                results.get("recall", float("nan")),
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

