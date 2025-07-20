#!/bin/bash
#PBS -lwalltime=06:00:00
#PBS -lselect=1:ncpus=16:mem=16gb


# ======= Runtime Environment =======
# Load modules or activate your Python env
# Example: conda

source ~/env_thesis/bin/activate
module load tools/prod
module load awscli

cd /rds/general/user/lrh24/home/thesis/Datasets/harvard_python

# ======= Launch your script =======
python download_script_2000_abnormal.py
python clean_data.py