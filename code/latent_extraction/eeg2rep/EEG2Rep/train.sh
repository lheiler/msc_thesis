#!/bin/bash
#PBS -lwalltime=06:00:00
#PBS -lselect=1:ncpus=16:mem=128gb:ngpus=1


cd /rds/general/user/lrh24/home/thesis/code/latent_extraction/eeg2rep/EEG2Rep

source ~/env_thesis/bin/activate

# Run your script
python main.py   --Training_mode Rep-Learning   --fif_root /rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/train   --fif_segment_len 10   --patch_size 16   --layers 4 --pre_layers 2   --emb_size 64 --num_heads 8 --dim_ff 256   --mask_ratio 0.55   --batch_size 32 --grad_accum_steps 8 --epochs 800   --lr 5e-4 --cov_weight 0.01 --early_stop_patience 0   --amp --warmup_epochs 10 --gpu 0