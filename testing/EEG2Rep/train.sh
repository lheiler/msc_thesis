#!/bin/bash
#SBATCH --job-name=EEG2Rep
#SBATCH --gres=gpu:1
#SBATCH --partition=AMD7-A100-T
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/testing/EEG2Rep

source /vol/bitbucket/lrh24/dlenv/bin/activate

nvidia-smi

# Run your script
python main.py --Training_mode Rep-Learning --fif_root /homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/train --batch_size 64 --epochs 1000  --gpu 0
