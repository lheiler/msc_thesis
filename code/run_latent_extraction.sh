#!/bin/bash
#SBATCH --job-name=latent_extraction
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/code

source /vol/bitbucket/lrh24/dlenv/bin/activate

# Run your script
python main.py

