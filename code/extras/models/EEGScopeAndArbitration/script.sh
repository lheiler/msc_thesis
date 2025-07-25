#!/bin/bash
#SBATCH --job-name=eegscope
#SBATCH --output=logs/eegscope_%j.out
#SBATCH --error=logs/eegscope_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Load your environment (if not already active)
source /vol/bitbucket/lrh24/dlenv/bin/activate
# Move to your code directory
cd ~/thesis/code/extras/models/EEGScopeAndArbitration

# Run the training script
python train_and_eval.py
