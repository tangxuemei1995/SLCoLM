#! /bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=relation
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=8gpu
#SBATCH --cpus-per-task=10
#SBATCH --time 72:00:00
python ./spert.py train --config configs/example_train.conf