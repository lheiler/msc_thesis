#!/bin/bash
#PBS -N final_eval
#PBS -lwalltime=24:00:00
#PBS -q v1_large24
#PBS -lselect=1:ncpus=64:mem=64gb

cd /rds/general/user/lrh24/home/thesis/code || exit

source ~/env_thesis/bin/activate

export PYTHONUNBUFFERED=1

LOGFILE="all_methods.log"
echo "=== Starting run at $(date) ===" | tee -a "$LOGFILE"

METHODS=(
  "jr_pc"
  "hopf_pc"
)

for METHOD in "${METHODS[@]}"; do
  echo -e "\n\n=== Running method: $METHOD at $(date) ===" | tee -a "$LOGFILE"
  python -u main.py --method "$METHOD" 2>&1 | tee -a "$LOGFILE"
done

echo -e "\n=== All runs completed at $(date) ===" | tee -a "$LOGFILE"
