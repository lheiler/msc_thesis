#!/bin/bash
#SBATCH --job-name=amortized_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/testing/amore/

source /vol/bitbucket/lrh24/dlenv/bin/activate

# Run your script
#python amortized_inference_post.py infer --psd_csv /homes/lrh24/thesis/testing/amore/empirical_psd.csv --out_dir results --n_samples 10000 
python amortized_inference_post.py infer --psd_csv /homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean --out_dir results --n_samples 1000 