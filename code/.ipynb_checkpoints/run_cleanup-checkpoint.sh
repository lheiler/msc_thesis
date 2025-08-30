#!/bin/bash
#PBS -N final_eval
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=16:ngpus=1:mem=32gb

cd /rds/general/user/lrh24/home/thesis/code

source ~/env_thesis/bin/activate


#python main.py --reset --method ctm_cma_avg
#python main.py --reset --method c22
#python main.py --reset --method jr_avg
#python main.py --reset --method hopf_avg
#python main.py --reset --method wong_wang_avg


#again




# change main back


#python main.py --method c22


#python main.py --method psd_ae_pc
#python main.py --method psd_ae_avg
python main.py --method eegnet

python main.py --method ctm_nn_pc
python main.py --method ctm_nn_avg

python main.py --method pca_pc
python main.py --method pca_avg
