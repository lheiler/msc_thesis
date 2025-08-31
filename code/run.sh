#!/bin/bash
#PBS -N final_eval
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=16:ngpus=1:mem=32gb

cd /rds/general/user/lrh24/home/thesis/code || exit

source ~/env_thesis/bin/activate

export PYTHONUNBUFFERED=1

LOGFILE="all_methods.log"
echo "=== Starting run at $(date) ===" | tee -a "$LOGFILE"

METHODS=(
  "c22"
)

for METHOD in "${METHODS[@]}"; do
  echo -e "\n\n=== Running method: $METHOD at $(date) ===" | tee -a "$LOGFILE"
  python -u main.py --method "$METHOD" 2>&1 | tee -a "$LOGFILE"
done

echo -e "\n=== All runs completed at $(date) ===" | tee -a "$LOGFILE"
