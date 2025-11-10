"""
MitoNet trainer implementation.

Wraps empanada's finetune functionality with the unified trainer interface.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob

from mitoem2.training.trainer import BaseTrainer
from mitoem2.models.mitonet.model import MitoNetModel
from mitoem2.configs import MitoNetConfig
from mitoem2.utils.logging import get_logger
from mitoem2.utils.paths import get_data_root
from mitoem2.data.converters.nnunet_converter import convert_nnunet_to_empanada_format
from mitoem2.evaluation.evaluate_res import evaluate_single_file

logger = get_logger(__name__)


class MitoNetTrainer(BaseTrainer):
    """
    MitoNet trainer that wraps empanada's finetune functionality.
    
    This trainer uses empanada's native training loop but provides
    a unified interface compatible with BaseTrainer.
    """

    def __init__(
        self,
        config: MitoNetConfig,
        model: Optional[MitoNetModel] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        use_empanada_finetune: bool = True,
    ):
        """
        Initialize MitoNet trainer.

        Args:
            config: MitoNet configuration.
            model: MitoNet model (optional, will be created from config if not provided).
            train_loader: Training data loader (optional, will be created from config if not provided).
            val_loader: Validation data loader (optional).
            use_empanada_finetune: Whether to use empanada's native finetune (recommended).
        """
        self.config = config
        self.use_empanada_finetune = use_empanada_finetune
        self.dataset_name = self._resolve_dataset_name()
        self.checkpoint_root = self._init_checkpoint_root()

        # Import empanada components from local source
        from mitoem2.utils.empanada_imports import (
            get_empanada_config_loaders,
            get_empanada_napari_finetune,
            get_empanada_napari_utils,
        )
        
        self.finetune_logic = get_empanada_napari_finetune()
        config_loaders = get_empanada_config_loaders()
        self.empanada_load_config = config_loaders.load_config
        utils_module = get_empanada_napari_utils()
        self.add_new_model = utils_module.add_new_model

        # For empanada finetune, we don't need the standard BaseTrainer setup
        # But we initialize it for compatibility
        if model is None:
            model = MitoNetModel(config_path=config.model.config_path)

        # Initialize BaseTrainer with dummy loaders (not used for empanada finetune)
        super().__init__(
            model=model,
            train_loader=train_loader or DataLoader([]),
            val_loader=val_loader,
            optimizer=None,  # empanada handles optimizer creation
            criterion=None,  # empanada handles loss
        )

    @classmethod
    def from_config(cls, config: MitoNetConfig) -> "MitoNetTrainer":
        """
        Create MitoNetTrainer from configuration.

        Args:
            config: MitoNet configuration.

        Returns:
            MitoNetTrainer instance.
        """
        return cls(config=config)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Note: This is not used when use_empanada_finetune=True.
        For empanada finetune, the training loop is handled by empanada.

        Args:
            batch: Batch of data.

        Returns:
            Dictionary of metrics for this step.
        """
        if self.use_empanada_finetune:
            raise NotImplementedError(
                "train_step is not used with empanada finetune. "
                "Use train() method which calls empanada's training loop."
            )
        
        # Placeholder for potential future custom training loop
        return {"loss": 0.0}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.

        Note: This is not used when use_empanada_finetune=True.

        Args:
            batch: Batch of data.

        Returns:
            Dictionary of metrics for this step.
        """
        if self.use_empanada_finetune:
            raise NotImplementedError(
                "validate_step is not used with empanada finetune."
            )
        
        # Placeholder for potential future custom validation loop
        return {"loss": 0.0}

    def _resolve_dataset_name(self) -> str:
        dataset_id = getattr(self.config.dataset, "id", None)
        if dataset_id is None:
            return "DatasetUnknown"
        try:
            from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
            dataset_name = maybe_convert_to_dataset_name(dataset_id)
            if dataset_name:
                return str(dataset_name)
        except ImportError:
            logger.warning("Could not import nnunetv2 utilities to resolve dataset name; using dataset id.")
        return str(dataset_id)

    def _init_checkpoint_root(self) -> Path:
        base_dir = Path(self.config.output.model_dir) if self.config.output.model_dir else Path("checkpoints") / "mitonet"
        checkpoint_root = base_dir / self.dataset_name
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        # Update config to reflect resolved model directory
        self.config.output.model_dir = str(checkpoint_root)
        return checkpoint_root

    def _prepare_empanada_config(self) -> Dict[str, Any]:
        """
        Prepare empanada configuration dictionary from MitoNetConfig.

        Returns:
            Configuration dictionary for empanada finetune.
        """
        # Load base finetune config
        finetune_config_path = Path(__file__).parent.parent.parent / "configs" / "mitonet_finetune_config.yaml"
        if not finetune_config_path.exists():
            # Try alternative location
            finetune_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "mitonet_finetune_config.yaml"
        
        if finetune_config_path.exists():
            empanada_config = self.empanada_load_config(str(finetune_config_path))
        else:
            logger.warning(f"Base finetune config not found at {finetune_config_path}, using defaults")
            empanada_config = self._get_default_empanada_config()

        # Load model config
        model_config_path = Path(self.config.model.config_path)
        if not model_config_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            model_config_path = Path(__file__).parent.parent / model_config_path
            print(f"model_config_path: {model_config_path}")
            if not model_config_path.exists():
                # Try configs directory
                model_config_path = project_root / "configs" / self.config.model.config_path

        if not model_config_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        model_config = self.empanada_load_config(str(model_config_path))

        # Merge configurations
        empanada_config['MODEL'] = {}
        for k, v in model_config.items():
            if k != 'FINETUNE':
                empanada_config['MODEL'][k] = model_config[k]
            else:
                empanada_config[k] = model_config[k]

        # Get dataset path
        dataset_id = self.config.dataset.id
        try:
            from nnunetv2.paths import nnUNet_raw
            from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
            dataset_name = maybe_convert_to_dataset_name(dataset_id)
            dataset_path = Path(nnUNet_raw) / dataset_name
        except ImportError:
            logger.warning("nnUNet not available, using dataset path from config")
            dataset_path = Path(self.config.dataset.root) if self.config.dataset.root else None
            if dataset_path is None:
                raise ValueError("Dataset path must be specified when nnUNet is not available")

        # Convert 3D to 2D slices if needed
        if self.config.output.output_data_path:
            output_data_path = Path(self.config.output.output_data_path)
        else:
            output_data_path = dataset_path / "2d_slices"

        skip_conversion = getattr(self.config, 'skip_conversion', False)
        if not skip_conversion:
            logger.info(f"Converting dataset to 2D slices: {dataset_path} -> {output_data_path}")
            convert_nnunet_to_empanada_format(
                nnunet_dataset_path=dataset_path,
                output_path=output_data_path,
            )
        else:
            logger.info(f"Using existing 2D slices at {output_data_path}")

        train_dir = output_data_path / "train"
        eval_dir = output_data_path / "val"

        # Update empanada config with training parameters
        empanada_config['model_name'] = self.config.model.model_name or "FinetunedMitoNet"
        empanada_config['TRAIN']['train_dir'] = str(train_dir)
        model_name = empanada_config['model_name']
        checkpoint_dir = self.checkpoint_root / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        empanada_config['TRAIN']['model_dir'] = str(checkpoint_dir)
        empanada_config['EVAL']['eval_dir'] = str(eval_dir)
        
        # Training parameters
        if hasattr(self.config, 'training'):
            training = self.config.training
            if isinstance(training, dict):
                empanada_config['TRAIN']['finetune_layer'] = training.get('finetune_layer', 'all')
                empanada_config['TRAIN']['batch_size'] = training.get('batch_size', 16)
                if 'schedule_params' in empanada_config['TRAIN']:
                    empanada_config['TRAIN']['schedule_params']['max_lr'] = training.get('learning_rate', 0.003)
                else:
                    empanada_config['TRAIN']['learning_rate'] = training.get('learning_rate', 0.003)
            else:
                empanada_config['TRAIN']['finetune_layer'] = getattr(training, 'finetune_layer', 'all')
                empanada_config['TRAIN']['batch_size'] = getattr(training, 'batch_size', 16)
                if 'schedule_params' in empanada_config['TRAIN']:
                    empanada_config['TRAIN']['schedule_params']['max_lr'] = getattr(training, 'learning_rate', 0.003)

        # Calculate epochs from iterations
        n_imgs = len(list(glob(str(train_dir / "**/images/*"), recursive=True)))
        if n_imgs == 0:
            raise ValueError(f"No training images found in {train_dir}")

        bsz = empanada_config['TRAIN']['batch_size']
        if n_imgs < bsz:
            logger.warning(f"Number of images ({n_imgs}) < batch size ({bsz}). Setting batch size to {n_imgs}.")
            empanada_config['TRAIN']['batch_size'] = n_imgs
            bsz = n_imgs

        iterations = getattr(self.config.training, 'iterations', 2000) if hasattr(self.config, 'training') else 2000
        epochs = int(iterations // (n_imgs // bsz)) if (n_imgs // bsz) > 0 else iterations
        logger.info(f"Found {n_imgs} images. Training for {epochs} epochs (~{iterations} iterations)")

        if 'schedule_params' in empanada_config['TRAIN']:
            empanada_config['TRAIN']['schedule_params']['epochs'] = epochs
        else:
            empanada_config['TRAIN']['epochs'] = epochs

        # Update patch size
        patch_size = getattr(self.config.training, 'patch_size', 256) if hasattr(self.config, 'training') else 256
        for aug in empanada_config['TRAIN']['augmentations']:
            for k in aug.keys():
                if ('height' in k or 'width' in k) and aug.get(k) is None:
                    aug[k] = patch_size

        # Save and eval frequency
        empanada_config['TRAIN']['save_freq'] = max(1, epochs // 5)
        empanada_config['EVAL']['epochs_per_eval'] = max(1, epochs // 5)

        # Update metrics labels
        for metric in empanada_config['TRAIN']['metrics'] + empanada_config['EVAL']['metrics']:
            if metric['metric'] in ['IoU', 'PQ']:
                metric['labels'] = empanada_config['MODEL']['labels']
            elif metric['metric'] in ['F1']:
                metric['labels'] = empanada_config['MODEL']['labels'][1:]  # Skip background

        return empanada_config

    def _get_default_empanada_config(self) -> Dict[str, Any]:
        """Get default empanada configuration."""
        return {
            'MODEL': {},
            'TRAIN': {
                'batch_size': 16,
                'finetune_layer': 'all',
                'schedule_params': {
                    'max_lr': 0.003,
                    'epochs': 100,
                },
                'augmentations': [
                    {'name': 'RandomFlip', 'p': 0.5},
                    {'name': 'RandomRotate90', 'p': 0.5},
                ],
                'metrics': [],
                'save_freq': 20,
            },
            'EVAL': {
                'eval_dir': None,
                'metrics': [],
                'epochs_per_eval': 20,
            },
            'FINETUNE': {},
        }

    def train(
        self,
        num_epochs: Optional[int] = None,
        start_epoch: int = 0,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """
        Train the model using empanada's finetune.

        Args:
            num_epochs: Number of epochs (overrides config if provided).
            start_epoch: Starting epoch (for resuming, not fully supported).
            checkpoint_path: Path to checkpoint to resume from (not fully supported).
        """
        if not self.use_empanada_finetune:
            # Fall back to base trainer
            super().train(num_epochs=num_epochs or 100, start_epoch=start_epoch, checkpoint_path=checkpoint_path)
            return

        logger.info("Starting MitoNet training using empanada finetune...")

        # Prepare empanada configuration
        empanada_config = self._prepare_empanada_config()

        # Override epochs if provided
        if num_epochs is not None:
            if 'schedule_params' in empanada_config['TRAIN']:
                empanada_config['TRAIN']['schedule_params']['epochs'] = num_epochs
            else:
                empanada_config['TRAIN']['epochs'] = num_epochs

        logger.info(f"Training configuration:")
        logger.info(f"  Model: {empanada_config.get('model_name', 'MitoNet')}")
        logger.info(f"  Train dir: {empanada_config['TRAIN']['train_dir']}")
        logger.info(f"  Model dir: {empanada_config['TRAIN']['model_dir']}")
        logger.info(f"  Batch size: {empanada_config['TRAIN']['batch_size']}")
        logger.info(f"  Epochs: {empanada_config['TRAIN'].get('schedule_params', {}).get('epochs', empanada_config['TRAIN'].get('epochs', 'N/A'))}")

        # Run empanada finetune
        try:
            # finetune.main is a function that takes config dict
            # self.finetune_logic is the finetune module
            self.finetune_logic.main(empanada_config)
            logger.info("MitoNet training completed successfully!")
        except Exception as e:
            logger.error(f"Error during MitoNet training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # Run test after training if requested
        test_after_training = getattr(self.config, 'test_after_training', False)
        if test_after_training:
            logger.info("Running test after training...")
            self._test_after_training(empanada_config)

    def _test_after_training(self, empanada_config: Dict[str, Any]) -> None:
        """Run inference on test set after training."""
        try:
            from mitoem2.inference.mitonet_inference import MitoNetInferenceEngine
            from mitoem2.models.mitonet.model import MitoNetModel
            import tifffile as tiff
            import numpy as np
            from glob import glob
            
            # Load trained model
            model_dir = Path(empanada_config['TRAIN']['model_dir'])
            model_config_path = model_dir / f"{empanada_config['model_name']}.yaml"
            
            if not model_config_path.exists():
                logger.warning(f"Trained model config not found: {model_config_path}")
                return

            logger.info(f"Loading trained model from {model_config_path}")
            model = MitoNetModel(config_path=str(model_config_path))
            
            # Load latest checkpoint if available
            checkpoint_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Loading checkpoint: {latest_checkpoint}")
                model.load_weights(latest_checkpoint)
            
            # Create inference engine with full configuration
            # Get axes from config
            axes = self.config.inference.axes if hasattr(self.config.inference, 'axes') else ["xy", "xz", "yz"]
            # Ensure axes is a list
            if isinstance(axes, str):
                axes = [axes]
            
            # Adjust pixel_vote_thr based on number of axes
            # If only one axis, pixel_vote_thr should be 1 (not 2)
            pixel_vote_thr = self.config.inference.pixel_vote_thr if hasattr(self.config.inference, 'pixel_vote_thr') else 2
            if len(axes) == 1:
                pixel_vote_thr = 1
                logger.info(f"Only one axis ({axes[0]}) specified, setting pixel_vote_thr to 1")
            elif pixel_vote_thr > len(axes):
                pixel_vote_thr = len(axes)
                logger.warning(f"pixel_vote_thr ({pixel_vote_thr}) > number of axes ({len(axes)}), setting to {len(axes)}")
            
            inference_config = {
                "downsampling": self.config.inference.downsampling,
                "confidence_thr": self.config.inference.confidence_thr,
                "center_confidence_thr": self.config.inference.center_confidence_thr,
                "min_distance_object_centers": self.config.inference.min_distance_object_centers,
                "min_size": self.config.inference.min_size,
                "min_extent": self.config.inference.min_extent,
                "use_gpu": self.config.inference.use_gpu,
                "axes": axes,
                "pixel_vote_thr": pixel_vote_thr,
            }
            
            engine = MitoNetInferenceEngine(model=model, config=inference_config)
            
            # Find test data directory
            dataset_id = self.config.dataset.id
            try:
                from nnunetv2.paths import nnUNet_raw
                from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
                dataset_name = maybe_convert_to_dataset_name(dataset_id)
                dataset_path = Path(nnUNet_raw) / dataset_name
            except ImportError:
                logger.warning("nnUNet not available, using dataset path from config")
                dataset_path = Path(self.config.dataset.root) if self.config.dataset.root else None
                if dataset_path is None:
                    logger.warning("Dataset path not available, skipping test")
                    return
            
            # Find test images
            test_images_dir = dataset_path / "imagesTs"
            if not test_images_dir.exists():
                logger.warning(f"Test images directory not found: {test_images_dir}")
                return
            
            # Find test image files
            test_image_files = []
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.nii.gz', '*.nii']:
                test_image_files.extend(glob(str(test_images_dir / ext)))
            
            if not test_image_files:
                logger.warning(f"No test images found in {test_images_dir}")
                return
            
            logger.info(f"Found {len(test_image_files)} test image(s)")
            
            # Create output directory
            if self.config.output.output_data_path:
                output_base = Path(self.config.output.output_data_path).parent
            else:
                output_base = dataset_path
            
            test_output_dir = output_base / f"imagesTs_{empanada_config['model_name']}_pred"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Test predictions will be saved to: {test_output_dir}")
            
            # Run inference on each test image
            consensus_output_files = []
            for test_image_file in test_image_files:
                test_image_path = Path(test_image_file)
                logger.info(f"Processing: {test_image_path.name}")
                
                try:
                    # Detect file format
                    file_ext = ''.join(test_image_path.suffixes) if test_image_path.suffixes else test_image_path.suffix
                    is_nifti = file_ext in [".nii.gz", ".nii"] or test_image_path.suffix == ".nii"
                    
                    # Load image and preserve metadata if NIfTI
                    sitk_image = None
                    if is_nifti:
                        import SimpleITK as sitk
                        sitk_image = sitk.ReadImage(str(test_image_path))
                        image = sitk.GetArrayFromImage(sitk_image)
                        # Store spacing and other metadata
                        spacing = sitk_image.GetSpacing()
                        origin = sitk_image.GetOrigin()
                        direction = sitk_image.GetDirection()
                    else:
                        image = tiff.imread(str(test_image_path))
                    
                    if image.ndim != 3:
                        logger.warning(f"Image {test_image_path.name} is not 3D, skipping")
                        continue
                    
                    logger.info(f"Image shape: {image.shape}, format: {file_ext}")
                    
                    # Run inference
                    segmentation = engine.predict(image, save_axis_results=False)
                    
                    # Save prediction in the same format as input
                    # Get base filename without extension (handle .nii.gz properly)
                    if is_nifti:
                        # For .nii.gz, stem only removes .gz, so we need to remove .nii as well
                        if file_ext == ".nii.gz":
                            base_filename = test_image_path.name.replace(".nii.gz", "")
                        else:
                            base_filename = test_image_path.stem
                    else:
                        base_filename = test_image_path.stem
                    
                    # Remove _0000 suffix if present (common in nnUNet format)
                    base_filename = base_filename.replace("_0000", "")
                    
                    # Save as NIfTI with preserved metadata
                    if is_nifti:
                        output_file = test_output_dir / f"{base_filename}_prediction{file_ext}"
                        # Convert segmentation to SimpleITK image
                        seg_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint16))
                        # Copy metadata from original image
                        seg_sitk.SetSpacing(spacing)
                        seg_sitk.SetOrigin(origin)
                        seg_sitk.SetDirection(direction)
                        # Write the image
                        sitk.WriteImage(seg_sitk, str(output_file))
                        logger.info(f"Saved prediction to {output_file} (NIfTI format with spacing: {spacing})")
                    else:
                        # Save as TIFF
                        output_file = test_output_dir / f"{base_filename}_prediction{file_ext}"
                        tiff.imwrite(str(output_file), segmentation.astype(np.uint16), compression='zlib')
                        logger.info(f"Saved prediction to {output_file} (TIFF format)")
                    
                    consensus_output_files.append(output_file)
                    
                except Exception as e:
                    logger.error(f"Error processing {test_image_path.name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Completed inference on {len(consensus_output_files)} test image(s)")
            
            # Run evaluation if ground truth is available
            gt_dir = dataset_path / "instancesTs"
            if not gt_dir.exists():
                gt_dir = dataset_path / "labelsTs"
            
            mask_dir = dataset_path / "masksTs"
            if not mask_dir.exists():
                mask_dir = None
                logger.warning("No mask directory found")  
            
            if gt_dir.exists():
                logger.info("Ground truth found, running evaluation...")
                self._evaluate_predictions(test_output_dir, gt_dir, mask_dir, consensus_output_files)
            else:
                logger.info("No ground truth found, skipping evaluation")

            
        except Exception as e:
            logger.error(f"Error during test after training: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _evaluate_predictions(
        self, 
        pred_dir: Path, 
        gt_dir: Path,
        mask_dir: Path,
        pred_files: List[Path]
    ) -> None:
        """Evaluate predictions against ground truth."""
        try:
            # Use the imported evaluate_single_file from mitoem2.evaluation
            # evaluate_single_file is already imported at the top of the file
            
            logger.info(f"Evaluating predictions in {pred_dir} against ground truth in {gt_dir}")
            
            # Create evaluation results directory
            eval_results_dir = pred_dir / "evaluation"
            eval_results_dir.mkdir(exist_ok=True)
            
            all_results = []
            
            # Match prediction files with ground truth files
            for pred_file in pred_files:
                # Get the format of the prediction file
                pred_ext = ''.join(pred_file.suffixes) if pred_file.suffixes else pred_file.suffix
                
                # Extract base name from prediction file (remove _prediction and extension)
                # Handle .nii.gz properly
                if pred_ext == ".nii.gz":
                    pred_name_without_ext = pred_file.name.replace("_prediction.nii.gz", "").replace(".nii.gz", "")
                elif pred_ext == ".nii":
                    pred_name_without_ext = pred_file.stem.replace("_prediction", "")
                else:
                    pred_name_without_ext = pred_file.stem.replace("_prediction", "")
                
                # Try to find matching ground truth file with the same format
                gt_file = None
                mask_file = None
                # First try the same format as prediction
                potential_gt = gt_dir / f"{pred_name_without_ext}{pred_ext}"
                if mask_dir:
                    potential_mask = mask_dir / f"{pred_name_without_ext}{pred_ext}"
                    if potential_mask.exists():
                        mask_file = potential_mask
                
                if potential_gt.exists():
                    gt_file = potential_gt
                else:
                    # If not found, try common alternatives
                    # For .nii.gz predictions, also try .nii
                    if pred_ext == ".nii.gz":
                        potential_gt = gt_dir / f"{pred_name_without_ext}.nii"
                        if potential_gt.exists():
                            gt_file = potential_gt
                    # For .nii predictions, also try .nii.gz
                    elif pred_ext == ".nii":
                        potential_gt = gt_dir / f"{pred_name_without_ext}.nii.gz"
                        if potential_gt.exists():
                            gt_file = potential_gt
                    # For TIFF predictions, try both .tiff and .tif
                    elif pred_ext in [".tiff", ".tif"]:
                        for alt_ext in ['.tiff', '.tif']:
                            if alt_ext != pred_ext:
                                potential_gt = gt_dir / f"{pred_name_without_ext}{alt_ext}"
                                if potential_gt.exists():
                                    gt_file = potential_gt
                                    break
                
                if gt_file is None:
                    logger.warning(f"No ground truth found for {pred_file.name} (looking for: {pred_name_without_ext}{pred_ext} or alternatives)")
                    continue
                
                logger.info(f"Evaluating {pred_file.name} against {gt_file.name}")
                
                try:
                    results = evaluate_single_file(
                        pred_file=str(pred_file),
                        gt_file=str(gt_file),
                        mask_file=mask_file,
                        save_results=True,
                    )
                    all_results.append({
                        'pred_file': str(pred_file),
                        'gt_file': str(gt_file),
                        'results': results
                    })
                    logger.info(f"Results accuracy: {results['accuracy']}, precision: {results['precision']}, recall: {results['recall']}, F1: {results['f1']}, binary accuracy: {results['binary_accuracy']}")
                except Exception as e:
                    logger.error(f"Error evaluating {pred_file.name}: {e}")
                    continue
            
            # Save summary
            if all_results:
                import json
                import numpy as np
                
                # Convert numpy types to Python native types for JSON serialization
                def convert_numpy_types(obj):
                    """Recursively convert numpy types to Python native types."""
                    # Handle numpy scalar types (int8, int16, int32, int64, uint8, uint16, uint32, uint64, etc.)
                    if isinstance(obj, (np.integer, np.unsignedinteger)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.complexfloating)):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    elif hasattr(obj, 'item'):  # Handle numpy scalar types that have .item() method
                        try:
                            return convert_numpy_types(obj.item())
                        except (ValueError, AttributeError):
                            return str(obj)  # Fallback to string if conversion fails
                    else:
                        return obj
                
                # Convert all results to JSON-serializable format
                serializable_results = convert_numpy_types(all_results)
                
                summary_file = eval_results_dir / "evaluation_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                logger.info(f"Evaluation summary saved to {summary_file}")
                
                # Calculate average metrics
                if len(all_results) > 0:
                    metrics_to_avg = ['PQ', 'SQ', 'RQ', 'IoU', 'F1']
                    avg_metrics = {}
                    for metric in metrics_to_avg:
                        values = []
                        for result in all_results:
                            if 'results' in result and metric in result['results']:
                                val = result['results'][metric]
                                # Convert numpy type to Python type
                                if isinstance(val, (np.integer, np.floating)):
                                    val = float(val) if isinstance(val, np.floating) else int(val)
                                values.append(val)
                        if values:
                            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
                    
                    if avg_metrics:
                        logger.info("Average metrics across all test images:")
                        for metric, value in avg_metrics.items():
                            logger.info(f"  {metric}: {value:.4f}")
            
        except ImportError:
            logger.warning("Evaluation module not available, skipping evaluation")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
