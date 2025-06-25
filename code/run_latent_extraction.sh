#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=32:mem=32gb

cd /rds/general/user/lrh24/home/thesis/code

# Load required modules
module load Python  # Replace with correct version

# Activate your virtual environment (if needed)
source ~/env_thesis/bin/activate

# Run your script
python main.py

