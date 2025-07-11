#!/bin/bash
#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=1:ngpus=1:mem=24gb
#PBS -o /rds/general/user/lrh24/home/thesis/code/job_output.log
#PBS -e /rds/general/user/lrh24/home/thesis/code/job_error.log

cd /rds/general/user/lrh24/home/thesis/code

# Load required modules

#module load Python  # Replace with correct version

module load tools/dev

# Activate your virtual environment (if needed)
source ~/env_thesis/bin/activate
#module load CUDA/12.1

#pip uninstall -y pycatch22
#pip install --no-cache-dir --no-binary=:all: pycatch22


# Run your script
python main.py

