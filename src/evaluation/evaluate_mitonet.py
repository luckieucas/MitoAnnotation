import os
import argparse
import numpy as np
import tifffile as tiff
from glob import glob
from pathlib import Path
import pandas as pd
from empanada.inference import Engine
from empanada.config_loaders import load_config

def load_model(model_config_path):
    """加载训练好的模型"""
    print(f"Loading model from {model_config_path}...")
    config = load_config(model_config_path)
    
    # 创建推理引擎
    engine = Engine(config)
    return engine

def predict_on_2d_slices(engine, image_dir, output_dir):
    """对2D切片进行预测"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted(glob(os.path.join(image_dir, '*.tif*')))
    print(f"Found {len(image_files)} images to predict")
    
    predictions = []
    for img_path in image_files:
        img = tiff.imread(img_path)
        base_name = os.path.basename(img_path)
        
        # 使用模型进行预测
        pred = engine.infer(img)
        
        # 保存预测结果
        output_path = os.path.join(output_dir, base_name)
        tiff.imwrite(output_path, pred, compression='zlib')
        
        predictions.append((base_name, pred))
        
        if len(predictions) % 10 == 0:
            print(f"  Processed {len(predictions)}/{len(image_files)} images")
    
    print(f"Predictions saved to {output_dir}")
    return predictions

def calculate_metrics(pred_dir, gt_dir, output_file=None):
    """计算评估指标"""
    print("Calculating evaluation metrics...")
    
    pred_files = sorted(glob(os.path.join(pred_dir, '*.tif*')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*.tif*')))
    
    if len(pred_files) != len(gt_files):
        print(f"Warning: Number of predictions ({len(pred_files)}) != number of ground truths ({len(gt_files)})")
    
    results = []
    
    for pred_path in pred_files:
        base_name = os.path.basename(pred_path)
        
        # 找到对应的ground truth
        gt_path = os.path.join(gt_dir, base_name)
        if not os.path.exists(gt_path):
            print(f"Warning: No ground truth found for {base_name}, skipping...")
            continue
        
        # 读取预测和ground truth
        pred = tiff.imread(pred_path)
        gt = tiff.imread(gt_path)
        
        if pred.shape != gt.shape:
            print(f"Warning: Shape mismatch for {base_name}: pred {pred.shape} vs gt {gt.shape}, skipping...")
            continue
        
        # 计算像素级别的指标
        # 假设这是instance segmentation，值>0表示前景
        pred_binary = (pred > 0).astype(np.uint8)
        gt_binary = (gt > 0).astype(np.uint8)
        
        # 计算IoU
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        iou = intersection / union if union > 0 else 0
        
        # 计算Dice coefficient
        dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0
        
        # 计算Precision和Recall
        tp = intersection
        fp = (pred_binary & ~gt_binary).sum()
        fn = (gt_binary & ~pred_binary).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'image': base_name,
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_pred_instances': len(np.unique(pred)) - 1,  # -1 for background
            'num_gt_instances': len(np.unique(gt)) - 1
        })
        
        if len(results) % 10 == 0:
            print(f"  Processed {len(results)} images")
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 计算平均指标
    mean_metrics = df[['iou', 'dice', 'precision', 'recall', 'f1']].mean()
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Number of images evaluated: {len(df)}")
    print(f"Mean IoU: {mean_metrics['iou']:.4f}")
    print(f"Mean Dice: {mean_metrics['dice']:.4f}")
    print(f"Mean Precision: {mean_metrics['precision']:.4f}")
    print(f"Mean Recall: {mean_metrics['recall']:.4f}")
    print(f"Mean F1: {mean_metrics['f1']:.4f}")
    print("="*50)
    
    # 保存详细结果
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
        
        # 保存汇总结果
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("="*50 + "\n")
            f.write("Evaluation Results Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Number of images evaluated: {len(df)}\n")
            f.write(f"Mean IoU: {mean_metrics['iou']:.4f}\n")
            f.write(f"Mean Dice: {mean_metrics['dice']:.4f}\n")
            f.write(f"Mean Precision: {mean_metrics['precision']:.4f}\n")
            f.write(f"Mean Recall: {mean_metrics['recall']:.4f}\n")
            f.write(f"Mean F1: {mean_metrics['f1']:.4f}\n")
            f.write("="*50 + "\n")
        print(f"Summary saved to {summary_file}")
    
    return df, mean_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate MitoNet model on test/validation data.")
    parser.add_argument("model_config", type=str, help="Path to the trained model config file (*.yaml).")
    parser.add_argument("image_dir", type=str, help="Directory containing test/validation images (2D slices).")
    parser.add_argument("gt_dir", type=str, help="Directory containing ground truth labels (2D slices).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save predictions. If not provided, will create next to image_dir.")
    parser.add_argument("--skip_prediction", action='store_true', help="Skip prediction and only calculate metrics from existing predictions.")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.image_dir), 'predictions')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 预测
    if not args.skip_prediction:
        engine = load_model(args.model_config)
        predictions = predict_on_2d_slices(engine, args.image_dir, args.output_dir)
    else:
        print("Skipping prediction, using existing predictions...")
    
    # 评估
    results_file = os.path.join(args.output_dir, 'evaluation_results.csv')
    df, mean_metrics = calculate_metrics(args.output_dir, args.gt_dir, results_file)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()



