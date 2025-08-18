#!/bin/bash
#PBS -lwalltime=03:00:00
#PBS -lselect=1:ncpus=32:mem=64gb

cd /rds/general/user/lrh24/home/thesis/code

source ~/env_thesis/bin/activate


#python main.py --reset --method ctm_cma_pc
python main.py --reset --method ctm_cma_avg

#python main.py --reset --method hopf_pc
python main.py --reset --method hopf_avg
python main.py --reset --method pca_pc
python main.py --reset --method pca_avg
python main.py --reset --method psd_ae_pc
python main.py --reset --method psd_ae_avg
python main.py --reset --method eegnet
#python main.py --reset --method eeg2rep
python main.py --reset --method c22
python main.py --reset --method ctm_nn_pc
python main.py --reset --method ctm_nn_avg
#python main.py --reset --method jr_pc
python main.py --reset --method jr_avg
#python main.py --reset --method wong_wang_pc
python main.py --reset --method wong_wang_avg