#!/usr/bin/env python3
"""
直接调用函数的nnunet训练示例脚本
不使用命令行参数，直接在代码中配置参数
"""

from auto_train_nnunet import AutoTrainNnunet
import logging

# 设置日志级别
logging.getLogger().setLevel(logging.INFO)

def main():
    """主函数：直接配置参数并运行训练"""
    
    # 配置参数
    DATA_FOLDER = "/projects/weilab/dataset/MitoLE/betaSeg"
    OUTPUT_DIR = "/projects/weilab/liupeng/MitoAnnotation/nnunet_data"
    DATASET_ID = "Dataset001_MitoLE"
    
    print("=== nnUNet自动训练配置 ===")
    print(f"数据文件夹: {DATA_FOLDER}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"数据集ID: {DATASET_ID}")
    print()
    
    # 创建自动训练实例
    auto_trainer = AutoTrainNnunet(
        data_folder=DATA_FOLDER,
        output_base_dir=OUTPUT_DIR,
        dataset_id=DATASET_ID
    )
    
    # 选择运行模式
    print("选择运行模式:")
    print("1. 完整流程 (转换数据 + 生成boundary + 训练 + 预测 + 后处理 + 评估)")
    print("2. 只转换数据和生成boundary")
    print("3. 只训练模型")
    print("4. 只运行预测和后处理")
    print("5. 自定义步骤")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == "1":
        print("\n运行完整流程...")
        auto_trainer.run_full_pipeline(
            fold=0,
            trainer="nnUNetTrainer",
            max_epochs=1000
        )
        
    elif choice == "2":
        print("\n只转换数据和生成boundary...")
        auto_trainer.convert_h5_to_tiff_direct()
        auto_trainer.generate_boundary_masks_direct()
        print("数据准备完成！")
        
    elif choice == "3":
        print("\n只训练模型...")
        # 确保数据已经准备好
        auto_trainer.convert_h5_to_tiff_direct()
        auto_trainer.generate_boundary_masks_direct()
        auto_trainer.nnunet_plan_and_process()
        auto_trainer.nnunet_train(fold=0, trainer="nnUNetTrainer", max_epochs=1000)
        
    elif choice == "4":
        print("\n只运行预测和后处理...")
        auto_trainer.nnunet_predict(fold=0, trainer="nnUNetTrainer")
        auto_trainer.postprocess_predictions()
        auto_trainer.evaluate_results()
        
    elif choice == "5":
        print("\n自定义步骤:")
        print("可用的步骤:")
        print("- convert_h5_to_tiff_direct()")
        print("- generate_boundary_masks_direct()")
        print("- nnunet_plan_and_process()")
        print("- nnunet_train(fold, trainer, max_epochs)")
        print("- nnunet_predict(fold, trainer)")
        print("- postprocess_predictions()")
        print("- evaluate_results()")
        
        # 这里可以添加自定义的逻辑
        print("请修改代码来实现自定义步骤")
        
    else:
        print("无效选择，退出程序")
        return
    
    print("\n=== 任务完成 ===")

if __name__ == "__main__":
    main()
