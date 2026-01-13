#!/bin/bash
#SBATCH --job-name=cwa-t_optuna
#SBATCH --gres=gpu:1
#SBATCH --partition=training
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/code/extras/models/CwA-T

source /vol/bitbucket/lrh24/dlenv/bin/activate

python optuna_tune.py configs/encoderL+transformer.yml --n_trials 30
