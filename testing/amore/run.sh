#!/bin/bash
#SBATCH --job-name=amortized_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G


cd /homes/lrh24/thesis/testing/amore/

source /vol/bitbucket/lrh24/dlenv/bin/activate

# Run your script
python amortized_inference_post.py train --num_sims 200000 --out_dir models/

