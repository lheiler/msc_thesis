#!/bin/bash
#SBATCH --job-name=cwa-t_optuna
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/code/extras/models/CwA-T

source /vol/bitbucket/lrh24/dlenv/bin/activate

python optuna_tuh.py --config configs/encoderS+transformer.yml --trials 100
