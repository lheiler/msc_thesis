#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=16:ngpus=1:mem=32gb



cd /rds/general/user/lrh24/home/thesis/code/extras/models/CwA-T

# Load required modules

#module load Python  # Replace with correct version

module load tools/dev

# Activate your virtual environment (if needed)
source ~/env_thesis/bin/activate
module load CUDA/12.1

#pip uninstall -y pycatch22
#pip install --no-cache-dir --no-binary=:all: pycatch22


# Run your script
python train_tuh.py configs/encoderL+transformer.yml 

