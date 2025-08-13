#!/bin/bash
#SBATCH --job-name=EEG2Rep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=16G


cd /homes/lrh24/thesis/testing/GC-VASE

source /vol/bitbucket/lrh24/dlenv/bin/activate

nvidia-smi

# Run your script 
python /homes/lrh24/thesis/testing/GC-VASE/gc_vase/train.py   --data_dir /homes/lrh24/thesis/testing/GC-VASE/   --data_line generic   --batch_size 256 --epochs 500 --lr 1e-4   --recon_enabled 1  --recon_weight 8 --scramble_permute_enabled 1   --conversion_permute_enabled 1
