#!/bin/bash
#PBS -lwalltime=04:00:00
#PBS -lselect=1:ncpus=16:mem=32gb:ngpus=1

cd /rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore

source ~/env_thesis/bin/activate

python amortized_inference_mlp.py
