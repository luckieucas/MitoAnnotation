#!/usr/bin/env python3
"""
自动训练nnunet的完整流程脚本
包含以下步骤：
1. 将h5文件转换为tiff格式并直接保存到nnunet数据集
2. 创建nnunet数据集
3. 生成带boundary的mask
4. nnunet plan and process
5. nnunet train
6. nnunet predict
7. 后处理（去掉boundary，bc_watershed）
8. 评估结果
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import logging
import time
import h5py
import tifffile as tiff
import numpy as np
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_train_nnunet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrainNnunet:
    def __init__(self, data_folder, output_base_dir, dataset_id="Dataset001_MitoLE"):
        """
        初始化自动训练nnunet类
        
        Args:
            data_folder: 包含_im.h5和_mito.h5文件的文件夹路径
            output_base_dir: 输出基础目录
            dataset_id: nnunet数据集ID
        """
        self.data_folder = Path(data_folder)
        self.output_base_dir = Path(output_base_dir)
        self.dataset_id = dataset_id
        self.dataset_number = str(int(dataset_id.split("_")[0].replace("Dataset", "")))
        
        # 创建必要的目录
        self.nnunet_dataset_dir = self.output_base_dir / "DATASET"
        self.nnUNet_raw_data_dir = self.nnunet_dataset_dir / "nnUNet_raw"
        self.nnUNet_preprocessed_dir = self.nnunet_dataset_dir / "nnUNet_preprocessed"
        self.nnUNet_results_dir = self.nnunet_dataset_dir / "nnUNet_trained_models"
        self.boundary_masks_dir = self.output_base_dir / "boundary_masks"
        
        # 确保输出目录存在
        for dir_path in [self.nnunet_dataset_dir, self.nnUNet_raw_data_dir, self.nnUNet_preprocessed_dir, self.nnUNet_results_dir, self.boundary_masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 创建nnunet数据集目录结构
        self.dataset_dir = self.nnUNet_raw_data_dir / self.dataset_id
        self.images_dir = self.dataset_dir / "imagesTr"
        self.labels_dir = self.dataset_dir / "labelsTr"
        self.instances_dir = self.dataset_dir / "instancesTr"
        self.images_ts_dir = self.dataset_dir / "imagesTs"
        self.instances_ts_dir = self.dataset_dir / "instancesTs"
        self.images_ts_pred_dir = self.dataset_dir / "imagesTs_pred"
        self.final_results_dir = self.dataset_dir / "imagesTs_pred_waterz"
        
        for dir_path in [self.images_dir, self.labels_dir, self.instances_dir, self.images_ts_dir, self.instances_ts_dir, self.images_ts_pred_dir, self.final_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


    def convert_h5_to_tiff_direct(self):
        """将h5文件转换为tiff格式并直接保存到nnunet数据集目录"""
        logger.info("开始转换h5文件到tiff格式并保存到nnunet数据集...")
        
        # 读取split.json文件
        split_json_path = self.data_folder / "split.json"
        if not split_json_path.exists():
            raise FileNotFoundError(f"未找到split.json文件: {split_json_path}")
        shutil.copy2(split_json_path, self.dataset_dir / "split.json")
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        
        train_files = set(split_data.get("train", []))
        test_files = set(split_data.get("test", []))
        
        logger.info(f"从split.json读取到 {len(train_files)} 个训练文件和 {len(test_files)} 个测试文件")
        logger.info(f"训练文件: {train_files}")
        logger.info(f"测试文件: {test_files}")
        
        # 查找所有h5文件
        h5_files = list(self.data_folder.glob("*.h5"))
        image_files = [f for f in h5_files if f.name.endswith('_im.h5')]
        label_files = [f for f in h5_files if f.name.endswith('_mito.h5')]
        
        logger.info(f"找到 {len(image_files)} 个图像文件和 {len(label_files)} 个标签文件")
        
        # 创建图像文件名到标签文件名的映射
        image_to_label_map = {}
        for img_file in image_files:
            # 从图像文件名提取基础名称（去掉_im.h5）
            base_name = img_file.stem.replace('_im', '')
            # 查找对应的标签文件
            label_file = self.data_folder / f"{base_name}_mito.h5"
            if label_file.exists():
                image_to_label_map[img_file] = label_file
                logger.info(f"找到匹配: {img_file.name} <-> {label_file.name}")
            else:
                logger.warning(f"未找到对应的标签文件: {img_file.name}")
        
        # 分离训练数据和测试数据
        training_pairs = []
        testing_pairs = []
        
        for img_file, label_file in image_to_label_map.items():
            try:
                # 获取原始文件名（去掉_im.h5后缀）
                base_name = img_file.stem.replace('_im', '')  # 例如: "high_c1"
                
                # 根据split.json判断是否为训练数据
                is_training = base_name in train_files
                is_testing = base_name in test_files
                
                # 如果文件既不在train也不在test中，跳过
                if not is_training and not is_testing:
                    logger.warning(f"文件 {base_name} 不在split.json的train或test列表中，跳过")
                    continue
                
                # 转换图像文件
                with h5py.File(img_file, 'r') as f:
                    if 'data' in f:
                        data = f['data'][()]
                    else:
                        first_key = list(f.keys())[0]
                        data = f[first_key][()]
                        logger.debug(f"使用数据集 '{first_key}'")
                    
                    # 归一化图像数据
                    if data.max() > data.min():
                        data = (data - data.min()) / (data.max() - data.min())
                        data = (data * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                    
                    # 使用nnunet格式的图像文件名（_0000.tiff结尾）
                    new_name = f"{base_name}_0000.tiff"
                    
                    if is_training:
                        # 训练数据保存到imagesTr目录
                        output_path = self.images_dir / new_name
                        logger.info(f"转换并保存训练图像: {img_file.name} -> {new_name}")
                    elif is_testing:
                        # 测试数据保存到imagesTs目录
                        output_path = self.images_ts_dir / new_name
                        logger.info(f"转换并保存测试图像: {img_file.name} -> {new_name}")
                    else:
                        # 这种情况不应该发生，因为前面已经过滤了
                        continue
                    
                    tiff.imwrite(output_path, data, compression="zlib")
                
                # 转换标签文件
                with h5py.File(label_file, 'r') as f:
                    if 'data' in f:
                        data = f['data'][()]
                    else:
                        first_key = list(f.keys())[0]
                        data = f[first_key][()]
                        logger.debug(f"使用数据集 '{first_key}'")
                    
                    # 确保标签数据是uint16类型
                    data = data.astype(np.uint16)
                    
                    # 使用基础名称保存标签（不加_0000后缀）
                    new_name = f"{base_name}.tiff"
                    
                    if is_training:
                        # 训练数据保存到labelsTr目录
                        output_path = self.instances_dir / new_name
                        logger.info(f"转换并保存训练标签: {label_file.name} -> {new_name}")
                        
                        # 添加到训练对列表
                        training_pairs.append({
                            "image": f"./imagesTr/{base_name}_0000.tiff",
                            "label": f"./labelsTr/{base_name}.tiff",
                            "original_image": img_file.name,
                            "original_label": label_file.name
                        })
                    elif is_testing:
                        # 测试数据保存到instancesTs目录（用于评估）
                        output_path = self.instances_ts_dir / new_name
                        logger.info(f"转换并保存测试标签: {label_file.name} -> {new_name}")
                        
                        # 添加到测试对列表
                        testing_pairs.append({
                            "image": f"./imagesTs/{base_name}_0000.tiff",
                            "label": f"./instancesTs/{base_name}.tiff",
                            "original_image": img_file.name,
                            "original_label": label_file.name
                        })
                    else:
                        # 这种情况不应该发生，因为前面已经过滤了
                        continue
                    
                    tiff.imwrite(output_path, data, compression="zlib")
                
            except Exception as e:
                logger.error(f"转换文件对失败: {img_file.name} + {label_file.name}: {e}")
                raise
        
        # 创建dataset.json文件
        dataset_json = {
            "name": self.dataset_id,
            "description": "MitoLE dataset for mitochondria segmentation",
            "reference": "Auto-generated for nnunet training",
            "licence": "Unknown",
            "release": "1.0",
            "tensorImageSize": "3D",
            "channel_names": {
                "0": "EM"
            },
            "labels": {
                "background": 0,
                "mitochondria": 1,
                "boundary": 2
            },
            "numTraining": len(training_pairs),
            "numTest": len(testing_pairs),
            "file_ending": ".tiff"
        }
        with open(self.dataset_dir / "dataset.json", 'w') as f:
            json.dump(dataset_json, f, indent=2)
        
        logger.info(f"nnunet数据集创建完成: {self.dataset_dir}")
        logger.info(f"成功转换 {len(training_pairs)} 个训练文件对")
        logger.info(f"成功转换 {len(testing_pairs)} 个测试文件对")
        
        # 打印文件对应关系
        logger.info("训练文件对:")
        for pair in training_pairs:
            logger.info(f"  {pair['original_image']} <-> {pair['original_label']}")
        
        logger.info("测试文件对:")
        for pair in testing_pairs:
            logger.info(f"  {pair['original_image']} <-> {pair['original_label']}")
        
    
    def generate_boundary_masks_direct(self):
        """生成带boundary的mask并直接保存到labelsTr目录（仅训练数据）"""
        logger.info("生成带boundary的mask（仅训练数据）...")
        
        try:
            # 导入必要的模块
            from connectomics.data.utils.data_segmentation import seg_to_instance_bd
            
            # 只处理训练标签文件（labelsTr目录）
            for label_file in self.instances_dir.glob("*.tiff"):
                try:
                    # 读取标签数据
                    vol = tiff.imread(label_file).astype(np.uint16)
                    logger.debug(f"处理训练标签文件: {label_file.name}, 唯一标签: {np.unique(vol)}")
                    
                    # 生成boundary mask
                    binary = (vol > 0).astype(np.uint8)
                    contour = seg_to_instance_bd(binary, tsz_h=3)
                    contour[contour > 0] = 2
                    
                    # 组合原始标签和boundary
                    saved_mask = binary + contour
                    saved_mask[saved_mask > 2] = 1
                    
                    # 保存到labelsTr目录
                    tiff.imwrite(self.labels_dir / label_file.name, saved_mask.astype(np.uint8), compression="zlib")
                    logger.info(f"更新训练标签文件: {label_file.name}")
                    
                except Exception as e:
                    logger.error(f"处理训练标签文件 {label_file.name} 失败: {e}")
                    raise
            
            logger.info("训练数据boundary mask生成完成")
            
        except ImportError as e:
            logger.error(f"导入connectomics模块失败: {e}")
            logger.info("尝试使用命令行方式生成boundary masks...")
            self.generate_boundary_masks_commandline()
        except Exception as e:
            logger.error(f"boundary mask生成失败: {e}")
            raise
    
    def generate_boundary_masks_commandline(self):
        """使用命令行方式生成boundary masks（备用方法）"""
        try:
            # 调用generate_contour.py脚本
            cmd = [
                sys.executable, "generate_contour.py",
                "-i", str(self.nnunet_dataset_dir / self.dataset_id / "labelsTr"),
                "-o", str(self.boundary_masks_dir),
                "-w", "3"  # boundary宽度
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("boundary mask生成完成（命令行方式）")
            logger.debug(f"生成输出: {result.stdout}")
            
            # 将生成的boundary masks复制回nnunet数据集
            labels_dir = self.nnUNet_raw_data_dir / self.dataset_id / "labelsTr"
            for boundary_file in self.boundary_masks_dir.glob("*.tiff"):
                # 找到对应的原始标签文件
                base_name = boundary_file.stem
                if base_name.startswith(self.dataset_id):
                    # 提取编号
                    parts = base_name.split("_")
                    if len(parts) >= 2:
                        number = parts[1]
                        target_name = f"{self.dataset_id}_{number}.tiff"
                        target_path = labels_dir / target_name
                        shutil.copy2(boundary_file, target_path)
                        logger.info(f"更新标签文件: {target_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"boundary mask生成失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def nnunet_plan_and_process(self):
        """运行nnunet plan and process"""
        logger.info("运行nnunet plan and process...")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['nnUNet_raw'] = str(self.nnUNet_raw_data_dir)
            env['nnUNet_preprocessed'] = str(self.nnUNet_preprocessed_dir)
            env['RESULTS_FOLDER'] = str(self.nnUNet_results_dir)
            
            # 确保这些目录存在
            for dir_path in [env['nnUNet_preprocessed'], env['RESULTS_FOLDER']]:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # 提取数据集ID中的数字部分
            
            # 运行nnunet plan - 实时显示输出
            plan_cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d", self.dataset_number,
                "--verify_dataset_integrity",
                "-c", "3d_fullres"
            ]
            
            # 移除 capture_output=True，实时显示输出
            result = subprocess.run(plan_cmd, env=env, check=True)
            logger.info("nnunet plan完成")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"nnunet plan失败: {e}")
            raise
    
    def nnunet_train(self, fold=0, trainer="nnUNetTrainer", max_epochs=1000):
        """运行nnunet训练"""
        logger.info(f"开始nnunet训练，fold: {fold}, trainer: {trainer}")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['nnUNet_raw'] = str(self.nnUNet_raw_data_dir)
            env['nnUNet_preprocessed'] = str(self.nnUNet_preprocessed_dir)
            env['RESULTS_FOLDER'] = str(self.nnUNet_results_dir)
            
            # 提取数据集ID中的数字部分
            
            # 运行训练
            train_cmd = [
                "nnUNetv2_train",
                self.dataset_number,
                "3d_fullres",
                str(fold)

            ]
            
            result = subprocess.run(train_cmd, env=env, check=True)
            logger.info("nnunet训练完成")
            logger.debug(f"训练输出: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"nnunet训练失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def nnunet_predict(self, fold=0, trainer="nnUNetTrainer"):
        """运行nnunet预测"""
        logger.info(f"开始nnunet预测，fold: {fold}, trainer: {trainer}")
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['nnUNet_raw'] = str(self.nnUNet_raw_data_dir)
            env['nnUNet_preprocessed'] = str(self.nnUNet_preprocessed_dir)
            env['RESULTS_FOLDER'] = str(self.nnUNet_results_dir)
            
            # 提取数据集ID中的数字部分
            
            # 运行预测
            predict_cmd = [
                "nnUNetv2_predict",
                "-i", str(self.nnUNet_raw_data_dir / self.dataset_id / "imagesTs"),
                "-o", str(self.images_ts_pred_dir),
                "-d", self.dataset_number,
                "-c", "3d_fullres",
                "-f", str(fold),
                "--save_probabilities"
            ]
            
            result = subprocess.run(predict_cmd, env=env, check=True)
            logger.info("nnunet预测完成")
            logger.debug(f"预测输出: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"nnunet预测失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def postprocess_predictions(self):
        """后处理预测结果：去掉boundary，运行bc_watershed"""
        logger.info("开始后处理预测结果...")
        
        try:
            # 运行bc_watershed.py
            cmd = [
                sys.executable, "./src/postprocessing/bc_watershed.py",
                "-i", str(self.images_ts_pred_dir),
                "-o", str(self.final_results_dir),
                "--save-tiff"
            ]
            
            result = subprocess.run(cmd, check=True)
            logger.info("后处理完成")
            logger.debug(f"后处理输出: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"后处理失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def evaluate_results(self, gt_folder=None):
        """评估最终结果"""
        logger.info("开始评估结果...")
        
        if gt_folder is None:
            # 使用测试标签作为ground truth
            gt_folder = self.instances_ts_dir
        
        try:
            # 对每个预测结果进行评估
            pred_files = list(self.final_results_dir.glob("*_seg.tiff"))
            
            for pred_file in pred_files:
                # 找到对应的ground truth文件
                base_name = pred_file.stem.replace("_seg", "")
                gt_file = gt_folder / f"{base_name}.tiff"
                
                if gt_file.exists():
                    logger.info(f"评估: {pred_file.name} vs {gt_file.name}")
                    
                    cmd = [
                        sys.executable, "./src/evaluation/evaluate_res.py",
                        "--pred_file", str(pred_file),
                        "--gt_file", str(gt_file)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    logger.info(f"评估完成: {pred_file.name}")
                    logger.debug(f"评估输出: {result.stdout}")
                else:
                    logger.warning(f"未找到对应的ground truth文件: {gt_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"评估失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise
    
    def run_full_pipeline(self, fold=0, trainer="nnUNetTrainer", max_epochs=1000, 
                         skip_convert=False, skip_boundary=False, skip_plan=False,
                         skip_training=False, skip_prediction=False, skip_postprocess=False, 
                         skip_evaluation=False):
        """运行完整的训练流程，每一步都可以选择执行或跳过"""
        logger.info("开始运行完整的nnunet训练流程...")
        start_time = time.time()
        
        try:
            # 1. 转换h5到tiff并直接保存到nnunet数据集
            if not skip_convert:
                logger.info("=== 步骤1: 转换h5到tiff格式 ===")
                self.convert_h5_to_tiff_direct()
            else:
                logger.info("=== 步骤1: 跳过h5到tiff转换 ===")
            
            # 2. 生成boundary masks（仅训练数据）
            if not skip_boundary:
                logger.info("=== 步骤2: 生成boundary masks ===")
                self.generate_boundary_masks_direct()
            else:
                logger.info("=== 步骤2: 跳过boundary masks生成 ===")
            
            # 3. nnunet plan and process
            if not skip_plan:
                logger.info("=== 步骤3: nnunet plan and process ===")
                self.nnunet_plan_and_process()
            else:
                logger.info("=== 步骤3: 跳过nnunet plan and process ===")
            
            # 4. nnunet训练
            if not skip_training:
                logger.info("=== 步骤4: nnunet训练 ===")
                self.nnunet_train(fold, trainer, max_epochs)
            else:
                logger.info("=== 步骤4: 跳过nnunet训练 ===")
            
            # 5. nnunet预测
            if not skip_prediction:
                logger.info("=== 步骤5: nnunet预测 ===")
                self.nnunet_predict(fold, trainer)
            
            # 6. 后处理
            if not skip_postprocess:
                logger.info("=== 步骤6: 后处理预测结果 ===")
                self.postprocess_predictions()
            else:
                logger.info("=== 步骤6: 跳过后处理 ===")
                
                # 如果跳过后处理，也跳过评估
                skip_evaluation = True
                logger.info("由于跳过后处理，自动跳过评估步骤")
            
            # 7. 评估结果
            if not skip_evaluation:
                logger.info("=== 步骤7: 评估结果 ===")
                self.evaluate_results()
            else:
                logger.info("=== 步骤7: 跳过评估 ===")
            
            total_time = time.time() - start_time
            logger.info(f"完整流程运行完成！总耗时: {total_time:.2f}秒")
            
            # 显示执行的步骤总结
            executed_steps = []
            if not skip_convert: executed_steps.append("h5转换")
            if not skip_boundary: executed_steps.append("boundary生成")
            if not skip_plan: executed_steps.append("nnunet规划")
            if not skip_training: executed_steps.append("nnunet训练")
            if not skip_prediction: executed_steps.append("nnunet预测")
            if not skip_postprocess: executed_steps.append("后处理")
            if not skip_evaluation: executed_steps.append("评估")
            
            logger.info(f"执行的步骤: {', '.join(executed_steps)}")
            
        except Exception as e:
            logger.error(f"流程运行失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="自动训练nnunet的完整流程")
    parser.add_argument("--data_folder", type=str, required=True,
                       help="包含_im.h5和_mito.h5文件的文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--dataset_id", type=str, default="Dataset001_MitoLE",
                       help="nnunet数据集ID")
    parser.add_argument("--fold", type=str, default="0",
                       help="训练fold编号")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer",
                       help="训练器名称")
    parser.add_argument("--max_epochs", type=int, default=1000,
                       help="最大训练轮数")
    
    # 每个步骤都可以选择跳过
    parser.add_argument("--skip_convert", action="store_true",
                       help="跳过h5到tiff转换步骤")
    parser.add_argument("--skip_boundary", action="store_true",
                       help="跳过boundary masks生成步骤")
    parser.add_argument("--skip_plan", action="store_true",
                       help="跳过nnunet plan and process步骤")
    parser.add_argument("--skip_training", action="store_true",
                       help="跳过nnunet训练步骤")
    parser.add_argument("--skip_prediction", action="store_true",
                       help="跳过nnunet预测步骤")
    parser.add_argument("--skip_postprocess", action="store_true",
                       help="跳过后处理步骤")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="跳过评估步骤")
    
    args = parser.parse_args()
    
    # 创建自动训练实例
    auto_trainer = AutoTrainNnunet(
        data_folder=args.data_folder,
        output_base_dir=args.output_dir,
        dataset_id=args.dataset_id
    )
    
    # 运行完整流程
    auto_trainer.run_full_pipeline(
        fold=args.fold,
        trainer=args.trainer,
        max_epochs=args.max_epochs,
        skip_convert=args.skip_convert,
        skip_boundary=args.skip_boundary,
        skip_plan=args.skip_plan,
        skip_training=args.skip_training,
        skip_prediction=args.skip_prediction,
        skip_postprocess=args.skip_postprocess,
        skip_evaluation=args.skip_evaluation
    )

if __name__ == "__main__":
    main()
