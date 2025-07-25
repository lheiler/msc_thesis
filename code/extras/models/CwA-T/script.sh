#!/bin/bash
#SBATCH --job-name=cwat_optuna
#SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output=/homes/lrh24/thesis/code/extras/models/CwA-T/logs/%x_%j.out
#SBATCH --error=/homes/lrh24/thesis/code/extras/models/CwA-T/logs/%x_%j.err

# (Optional) load modules if your cluster needs them
# module load cuda/11.8
# module load python/3.10

cd /homes/lrh24/thesis/code/extras/models/CwA-T/

# Activate your virtual environment
source /vol/bitbucket/lrh24/dlenv/bin/activate

# Use srun to launch the task under Slurm
python optuna_tune.py configs/encoderL+transformer.yml --n_trials 50
