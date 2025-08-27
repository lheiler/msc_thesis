#!/bin/bash
#PBS -lwalltime=04:00:00
#PBS -lselect=1:ncpus=64:mem=128gb

cd /rds/general/user/lrh24/home/thesis/code/utils

source ~/env_thesis/bin/activate

python cleanup_real_eeg_tuh.py