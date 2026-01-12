#!/bin/bash
#PBS -qv1_medium72
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=32:mem=64gb


# ======= Runtime Environment =======
# Load modules or activate your Python env
# Example: conda

module load tools/prod
module load awscli

source ~/env_thesis/bin/activate

cd /rds/general/user/lrh24/home/thesis/Datasets/harvard_python

# ======= Launch your script =======
python download_script_500_age.py
