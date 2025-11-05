import argparse
import subprocess
import sys
import os
from pathlib import Path

def create_slurm_script(
    command, 
    job_name, 
    partition, 
    time, 
    cpus, 
    mem, 
    gpus, 
    constraint,
    env_name,
    email,
    node_type
):
    """根据传入的参数动态创建SLURM sbatch脚本内容"""
    
    # 如果用户没有提供job_name，就根据command自动生成一个
    if not job_name:
        job_name = f"{command.replace(' ', '_')}"

    # 创建日志目录（如果不存在）
    Path("logs").mkdir(exist_ok=True)

    # 根据节点类型构建GPU相关配置
    gpu_config = ""
    if node_type == "gpu":
        gpu_config = f"#SBATCH --gres=gpu:{gpus}"
        if constraint:
            gpu_config += f'\n#SBATCH --constraint="{constraint}"'

    # 使用f-string构建脚本模板
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{gpu_config}
#SBATCH --output=logs/%j_{job_name}.out
#SBATCH --error=logs/%j_{job_name}.err
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications (on job start, end, or fail)
#SBATCH --mail-user={email}  # Email address for notifications

# 切换到工作目录
cd {os.getcwd()}

echo "=========================================================="
echo "Starting on $(hostname)"
echo "Time is $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node type: {node_type.upper()}"
echo "Running command: {command}"
echo "=========================================================="

# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {env_name}

# 验证环境激活
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# 设置 just runtime directory to avoid permission errors
# Try user's home directory first, fallback to project directory
if [ -n "$HOME" ] && [ -w "$HOME" ]; then
    export JUST_RUNTIME_DIR="$HOME/.just_runtime"
else
    export JUST_RUNTIME_DIR="{os.getcwd()}/.just_runtime"
fi
mkdir -p "$JUST_RUNTIME_DIR" 2>/dev/null || true
chmod 700 "$JUST_RUNTIME_DIR" 2>/dev/null || true
echo "JUST_RUNTIME_DIR set to: $JUST_RUNTIME_DIR"

# 运行 脚本
{command}

# 任务结束后发送 discord 通知
if command -v discord &> /dev/null; then
    discord "command '{command}' finished on job $SLURM_JOB_ID"
else
    echo "Discord command not found, skipping notification"
fi
"""
    return slurm_script

def submit_job(args):
    """提交SLURM作业"""
    
    # 1. 创建SLURM脚本内容
    script_content = create_slurm_script(
        command=args.command,
        job_name=args.job_name,
        partition=args.partition,
        time=args.time,
        cpus=args.cpus,
        mem=args.mem,
        gpus=args.gpus,
        constraint=args.constraint,
        env_name=args.env_name,
        email=args.email,
        node_type=args.node_type
    )
    
    print("--- 将要提交的SLURM脚本 ---")
    print(script_content)
    print("--------------------------")
    
    try:
        # 2. 通过stdin将脚本内容传递给sbatch，并执行
        result = subprocess.run(
            ['sbatch'],
            input=script_content,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ 作业提交成功!")
        print(f"   {result.stdout.strip()}")

    except FileNotFoundError:
        print("❌ 错误: 'sbatch' 命令未找到。请确保 Slurm 环境已正确配置。", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("❌ 错误: 作业提交失败。", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="将 python 命令作为 SLURM 作业提交。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 显示默认值
    )
    
    # --- 必选参数 ---
    parser.add_argument('--command', help='要运行的 python 命令 (例如: train_model)')
    
    # --- 可选参数 (覆盖SLURM默认配置) ---
    parser.add_argument('--node_type', default='gpu', choices=['gpu', 'cpu'], 
                        help='节点类型: gpu (使用GPU节点) 或 cpu (使用CPU节点)')
    parser.add_argument('--job_name', '-n', help='指定作业的名称 (默认会根据命令自动生成)')
    parser.add_argument('--time', '-t', default='120:00:00', help='作业运行时长 (D-HH:MM:SS)')
    parser.add_argument('--partition', '-p', default='long', help='指定要提交到的分区')
    parser.add_argument('--cpus', '-c', type=int, default=32, help='每个任务请求的CPU核心数')
    parser.add_argument('--mem', '-m', default='240G', help='请求的内存大小 (例如: 240G)')
    parser.add_argument('--gpus', '-g', type=int, default=1, help='请求的GPU数量 (仅在node_type=gpu时生效)')
    parser.add_argument('--constraint', default='', help='GPU类型约束 (例如: "v100|a100", 仅在node_type=gpu时生效)')
    parser.add_argument('--env_name', '-e', default='mitoem2', help='指定要激活的conda环境')
    parser.add_argument('--email',default='liupen@bc.edu', help='指定要发送邮件的邮箱')
    args = parser.parse_args()
    

    submit_job(args)

if __name__ == '__main__':
    main()