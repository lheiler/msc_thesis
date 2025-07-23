#!/bin/bash
#PBS -lwalltime=10:00:00
#PBS -lselect=1:ncpus=32:mem=32gb


# ======= Runtime Environment =======
# Load modules or activate your Python env
# Example: conda

source ~/env_thesis/bin/activate

cd /rds/general/user/lrh24/home/thesis/Datasets/harvard_python

# ======= Launch your script =======
python clean_data.py
