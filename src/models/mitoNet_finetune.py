import os
import argparse
import platform
import torch
from glob import glob
from empanada.config_loaders import load_config
from empanada_napari.utils import add_new_model

# 导入核心训练逻辑
from empanada_napari import finetune as finetune_logic

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with custom data.")
    parser.add_argument("train_dir", type=str, help="Path to the training data directory.")
    parser.add_argument("model_dir", type=str, help="Directory to save the fine-tuned model.")
    parser.add_argument("--eval_dir", type=str, default=None, help="Path to the validation data directory (optional).")
    parser.add_argument("--model_config", type=str, default="MitoNet_v1.yaml", help="Path to the base model config file to fine-tune.")
    parser.add_argument("--model_name", type=str, default="FinetunedMitoNet", help="Name for the new fine-tuned model.")
    parser.add_argument("--iterations", type=int, default=500, help="Number of training iterations.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size in pixels for training.")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Maximum learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--finetune_layer", type=str, default='all', choices=['none', 'stage4', 'stage3', 'stage2', 'stage1', 'all'], help="Encoder layers to finetune.")
    
    args = parser.parse_args()

    print("Loading base finetune configuration...")
    # 从 empanada_napari 加载默认的微调配置
    # 这可以通过找到 finetune_config.yaml 的路径来实现
    import empanada_napari
    base_dir = os.path.dirname(empanada_napari.__file__)
    main_config_path = os.path.join(base_dir, 'training/finetune_config.yaml')
    config = load_config(main_config_path)

    print(f"Loading model-specific config from {args.model_config}...")
    model_config = load_config(args.model_config)

    # --- 合并和覆盖配置 ---
    # 1. 将模型定义（MODEL）和微调参数（FINETUNE）从模型特定配置中加载
    config['MODEL'] = {}
    for k, v in model_config.items():
        if k != 'FINETUNE':
            config['MODEL'][k] = model_config[k]
        else:
            config[k] = model_config[k]

    # 2. 设置用户自定义的参数
    config['model_name'] = args.model_name
    config['TRAIN']['train_dir'] = args.train_dir
    config['TRAIN']['model_dir'] = args.model_dir
    config['EVAL']['eval_dir'] = args.eval_dir
    config['TRAIN']['finetune_layer'] = args.finetune_layer
    config['TRAIN']['batch_size'] = args.batch_size
    config['TRAIN']['schedule_params']['max_lr'] = args.learning_rate

    # 3. 根据图像数量和迭代次数计算epochs
    n_imgs = len(glob(os.path.join(args.train_dir, '**/images/*')))
    if not n_imgs:
        raise Exception(f"No images found in {os.path.join(args.train_dir, '/images/*')}")
    
    bsz = config['TRAIN']['batch_size']
    if n_imgs < bsz:
        print(f"Warning: Number of images ({n_imgs}) is less than batch size ({bsz}). Setting batch size to {n_imgs}.")
        config['TRAIN']['batch_size'] = n_imgs
        bsz = n_imgs

    epochs = int(args.iterations // (n_imgs // bsz)) if (n_imgs // bsz) > 0 else args.iterations
    print(f"Found {n_imgs} images for training. Training for {epochs} epochs to approximate {args.iterations} iterations.")

    # 4. 更新配置中的 epochs 和 patch size
    if 'epochs' in config['TRAIN']['schedule_params']:
        config['TRAIN']['schedule_params']['epochs'] = epochs
    else:
        config['TRAIN']['epochs'] = epochs

    for aug in config['TRAIN']['augmentations']:
        for k in aug.keys():
            if ('height' in k or 'width' in k) and aug.get(k) is None:
                 aug[k] = args.patch_size

    config['TRAIN']['save_freq'] = max(1, epochs // 5)
    config['EVAL']['epochs_per_eval'] = max(1, epochs // 5) if args.eval_dir else epochs + 1

    # 5. 填充度量（metrics）所需的标签
    for metric in config['TRAIN']['metrics'] + config['EVAL']['metrics']:
        if metric['metric'] in ['IoU', 'PQ']:
            metric['labels'] = config['MODEL']['labels']
        elif metric['metric'] in ['F1']:
            metric['labels'] = config['MODEL']['thing_list']

    # 6. 处理平台特定问题
    if platform.system() == 'Darwin':
        config['TRAIN']['workers'] = 0

    # --- 开始微调 ---
    print("Starting fine-tuning process...")
    finetune_logic.main(config)
    print("Fine-tuning finished!")

    # --- 注册新模型 ---
    # 微调后，empanada 会保存一个新的 .pth 文件和 .yaml 文件
    # 我们可以将这个新生成的 yaml 文件注册到 empanada 中，方便之后调用
    output_model_yaml = os.path.join(args.model_dir, args.model_name + '.yaml')
    if os.path.exists(output_model_yaml):
        print(f"Registering new model '{args.model_name}'...")
        add_new_model(args.model_name, output_model_yaml)
        print("Model registered successfully. You can now use it in the prediction script.")
    else:
        print("Could not find the output model yaml. Skipping registration.")

if __name__ == "__main__":
    main()
