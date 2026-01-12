#!/bin/bash
#PBS -N download_sleep_edf
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=8:mem=16gb

cd /rds/general/user/lrh24/ephemeral/sleep_edf

source ~/env_thesis/bin/activate


#sleep-edf
wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/


