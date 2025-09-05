#!/usr/bin/env python3
"""
简单的nnunet训练示例
直接调用函数，无需命令行参数
"""

from auto_train_nnunet import AutoTrainNnunet

def run_simple_training():
    """运行简单的训练流程"""
    
    # 配置参数
    data_folder = "/projects/weilab/dataset/MitoLE/betaSeg"
    output_dir = "/projects/weilab/liupeng/MitoAnnotation/nnunet_data"
    dataset_id = "Dataset001_MitoLE"
    
    print(f"开始训练，数据文件夹: {data_folder}")
    print(f"输出目录: {output_dir}")
    
    # 创建训练器实例
    trainer = AutoTrainNnunet(
        data_folder=data_folder,
        output_base_dir=output_dir,
        dataset_id=dataset_id
    )
    
    # 运行完整流程
    trainer.run_full_pipeline(
        fold=0,
        trainer="nnUNetTrainer",
        max_epochs=1000
    )

def run_step_by_step():
    """分步骤运行训练流程"""
    
    data_folder = "/projects/weilab/dataset/MitoLE/betaSeg"
    output_dir = "/projects/weilab/liupeng/MitoAnnotation/nnunet_data"
    dataset_id = "Dataset001_MitoLE"
    
    trainer = AutoTrainNnunet(
        data_folder=data_folder,
        output_base_dir=output_dir,
        dataset_id=dataset_id
    )
    
    print("步骤1: 转换数据并创建数据集...")
    trainer.convert_h5_to_tiff_direct()
    
    print("步骤2: 生成boundary masks...")
    trainer.generate_boundary_masks_direct()
    
    print("步骤3: 运行nnunet plan and process...")
    trainer.nnunet_plan_and_process()
    
    print("步骤4: 开始训练...")
    trainer.nnunet_train(fold=0, trainer="nnUNetTrainer", max_epochs=1000)
    
    print("步骤5: 运行预测...")
    trainer.nnunet_predict(fold=0, trainer="nnUNetTrainer")
    
    print("步骤6: 后处理...")
    trainer.postprocess_predictions()
    
    print("步骤7: 评估结果...")
    trainer.evaluate_results()
    
    print("所有步骤完成！")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 完整流程")
    print("2. 分步骤运行")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        run_simple_training()
    elif choice == "2":
        run_step_by_step()
    else:
        print("无效选择")
