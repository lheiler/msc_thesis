#!/bin/bash
#PBS -NconvAE_train
#PBS -lwalltime=12:00:00
#PBS -lselect=1:ncpus=4:mem=32gb

                             # (a)bort, (b)egin, (e)nd mail
# ===== load software stack =====
#module purge
#module load cuda/12.2                   # GPU drivers + nvcc
#module load anaconda/2024.02            # Python distro (example)

# ===== activate your virtual/conda env =====
source env_thesis/bin/activate              # or: conda activate env_thesis

cd /rds/general/user/lrh24/home/thesis/code/Latent_Extraction/AE

# ===== go to the directory from which you ran qsub =====
#cd $PBS_O_WORKDIR                       # preserves relative paths

# (Optional) show resources for debugging

#echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"
#nvidia-smi                                 # quick sanity check

# ===== run your script =====
python convAE.py
