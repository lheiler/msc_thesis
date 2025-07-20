#!/bin/bash
#PBS -lwalltime=02:00:00              
#PBS -lselect=1:ncpus=32:ngpus=1:mem=32gb     
#PBS -o /rds/general/user/lrh24/home/thesis/code/job_output_all.log
#PBS -e /rds/general/user/lrh24/home/thesis/code/job_error_all.log

# ------------------------------------------------------------------
# Change to project root (where main.py lives)
# ------------------------------------------------------------------
cd /rds/general/user/lrh24/home/thesis/code

# ------------------------------------------------------------------
# Load modules / virtual-env  (same as run_latent_extraction.sh)
# ------------------------------------------------------------------
module load tools/prod
module load tools/dev
source ~/env_thesis/bin/activate    # activate your Python env
module load CUDA/12.0.0              # enable if GPU jobs


pip uninstall -y pycatch22
pip install --no-cache-dir --no-binary=:all: pycatch22

# ------------------------------------------------------------------
# Run pipeline for each config file
# ------------------------------------------------------------------
CONFIGS=(
  "bids_500_normal_abnormal_clean_ctm.yaml"
  "bids_500_normal_abnormal_clean_c22.yaml"
  "bids_500_normal_abnormal_clean_AE.yaml"
  "bids_2000_normal_abnormal_clean_ctm.yaml"
  "bids_2000_normal_abnormal_clean_c22.yaml"
  "bids_2000_normal_abnormal_clean_AE.yaml"
  "bids_age_500_clean_c22.yaml"
  "bids_age_500_clean_AE.yaml"
  "bids_age_500_clean_ctm.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  #echo "🚀 Running $cfg at $(date)" | tee -a run_all_configs.log
  python main.py --config "configs/${cfg}"
  status=$?
  #echo "✔️  Finished $cfg with exit code $status at $(date)" | tee -a run_all_configs.log
  if [[ $status -ne 0 ]]; then
    echo "⚠️  Pipeline failed for $cfg; aborting." >&2
    exit $status
  fi
  #echo "---------------------------------------------" | tee -a run_all_configs.log
done

#echo "✅ All configs completed successfully." | tee -a run_all_configs.log 
