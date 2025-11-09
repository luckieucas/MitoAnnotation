#!/usr/bin/env python
"""
Unified evaluation script for mitoem2.
"""
import argparse
from pathlib import Path

from mitoem2.utils.logging import setup_logger


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate mitoem2 predictions")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to prediction file or directory",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to ground truth file or directory",
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="Optional mask file for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU threshold for instance matching",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(name="mitoem2_evaluate")

    # Import evaluation function
    # Note: This uses the existing evaluation module from src/evaluation
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "src" / "evaluation"))
    
    try:
        from evaluate_res import evaluate_single_file, evaluate_directory
    except ImportError:
        logger.error("Could not import evaluation module. Please ensure src/evaluation is available.")
        return

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    # Evaluate
    if pred_path.is_file() and gt_path.is_file():
        # Single file evaluation
        logger.info(f"Evaluating {pred_path} against {gt_path}")
        results = evaluate_single_file(
            pred_file=str(pred_path),
            gt_file=str(gt_path),
            mask_file=str(args.mask) if args.mask else None,
            save_results=True,
        )
        logger.info(f"Results: {results}")
    elif pred_path.is_dir() and gt_path.is_dir():
        # Directory evaluation
        logger.info(f"Evaluating directory {pred_path} against {gt_path}")
        # Use evaluate_directory if available
        logger.info("Directory evaluation not yet implemented in unified script")
    else:
        logger.error("Prediction and ground truth must both be files or both be directories")
        return

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
