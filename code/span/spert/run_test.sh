#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=relation
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 72:00:00
#需要修改example_eval.conf中的模型地址
python spert.py eval --config configs/example_eval.conf