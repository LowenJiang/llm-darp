#!/bin/bash
#SBATCH --job-name=llm_or
#SBATCH --partition=cornell
#SBATCH --account=cornell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1  # Request 1 GPU (max 8)
#SBATCH --time=24:00:00     # Time limit hrs:min:sec
#SBATCH --output=job_%j.out
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

# Run your command
srun python3 src/meta_train.py --solver-ckpt checkpoints/Feb_01_6_finetune/latest.pt --episodes 25000 --num-envs 500 --device cuda

