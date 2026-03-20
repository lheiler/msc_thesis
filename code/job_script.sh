#!/bin/bash
#PBS -N final_eval
#PBS -q v1_large24
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=64:mem=128gb


# cd /rds/general/user/lrh24/home/msc_thesis/code

# source ~/env_thesis/bin/activate

# python main.py --method ctm_nn_pc
# python main.py --method ctm_nn_avg
# python main.py --method pca_pc
# python main.py --method pca_avg
# python main.py --method eegnet
# python main.py --method psd_ae_avg
# python main.py --method psd_ae_pc

# python main.py --method c22
python main.py --method jr_avg
python main.py --method jr_pc
python main.py --method wong_wang_avg
python main.py --method hopf_avg
python main.py --method hopf_pc
python main.py --method ctm_cma_avg


