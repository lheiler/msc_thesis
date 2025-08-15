#!/bin/bash
#SBATCH --job-name=EEG2Rep
#SBATCH --gres=gpu:1
#SBATCH --partition=gpgpuB
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G


cd /homes/lrh24/thesis/code/latent_extraction/gc_vase/GC-VASE/

source /vol/bitbucket/lrh24/dlenv/bin/activate

nvidia-smi

# Run your script 
python /homes/lrh24/thesis/code/latent_extraction/gc_vase/GC-VASE/gc_vase/train.py  --data_dir /homes/lrh24/thesis/code/latent_extraction/gc_vase/GC-VASE/   --data_line generic   --batch_size 256 --epochs 500 --lr 1e-4  --num_layers 4 --recon_enabled 1  --recon_weight 1 --scramble_permute_enabled 1   --conversion_permute_enabled 1
