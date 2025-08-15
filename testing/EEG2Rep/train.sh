#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=16:mem=128gb:ngpus=1:gpu_type=A100


cd /rds/general/user/lrh24/home/thesis/testing/EEG2Rep

source /rds/general/user/lrh24/home/thesis/testing/EEG2Rep/rep_env/bin/activate

# Run your script
python main.py --Training_mode Rep-Learning --fif_root /rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/train --fif_segment_len 10 --batch_size 128 --epochs 1000 --emb_size 128 --layers 8 --pre_layers 4 --num_heads 8 --dim_ff 512 --gpu 0
