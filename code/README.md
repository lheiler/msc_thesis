## EEG Latent-Feature Pipeline (TUH-focused)

Research-grade, end-to-end pipeline for EEG latent-feature extraction and downstream evaluation/classification. The current code path is TUH-centric and driven by a simple YAML config. Cached latent features are reused between runs unless reset.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Preprocessing (TUH EDF → cleaned)](#preprocessing)
5. [Data expectations](#data-expectations)
6. [Configuration](#configuration)
7. [Running](#running)
8. [Outputs](#outputs)
9. [Extraction methods](#extraction-methods)
10. [HPC usage](#hpc-usage)
11. [Troubleshooting](#troubleshooting)

---

## Features
- **Modular pipeline**: data loading → latent extraction → Optuna search → evaluation → reports.
- **Many extraction options**: mechanistic models (CTM/JR/Wong–Wang/Hopf), statistical (Catch22, PCA), and learned (EEGNet-AE, EEG2Rep, PSD-AE).
- **Config/CLI driven**: choose dataset root, method, and optimisation knobs via YAML/flags.
- **Caching**: latent features written as JSONL and reused on subsequent runs.
- **Reproducible reports**: text, markdown, JSON, and figures per run under `Results/`.

---

## Project Structure
```
code/
├── data_preprocessing/      # TUH .fif loader
├── latent_extraction/       # All extractors and models
├── model_training/          # Optuna search + single-task head
├── evaluation/              # Latent metrics, plots, reporting
├── utils/                   # Cleaning, PSD, dataset utilities
├── Results/                 # Auto-generated per-run outputs
├── configs/                 # Example (legacy) configs
├── main.py                  # Entry point
├── run_all_configs.sh       # Batch runner (PBS example)
├── run_cleanup.sh           # Force re-extraction across methods
├── run_latent_extraction.sh # SLURM example (update paths before use)
├── requirements.txt
└── README.md
```

Note: some configs under `configs/` use a legacy schema and may not match the current loader (see Configuration below).

---

## Setup
```bash
# Optional: create and activate a venv
python -m venv ~/env_thesis && source ~/env_thesis/bin/activate

# Install dependencies
pip install -r requirements.txt

# On some clusters, compile pycatch22 from source (see HPC usage)
```

---

## Preprocessing

If you start from TUH EDFs, run the cleaning/export utility first to produce cleaned, standardised data. This script performs channel renaming, bad-channel interpolation, trimming zero edges, notch filtering at mains and harmonics, ICA (EOG/ECG), rereferencing, low-pass, artifact annotations, canonical 19‑channel ordering, epoching, AutoReject, basic QC, and per-epoch z-scoring.

```bash
python -m utils.cleanup_real_eeg_tuh \
  # or open and run the __main__ example at the bottom of utils/cleanup_real_eeg_tuh.py
```

Programmatic usage (example):
```python
from utils.cleanup_real_eeg_tuh import load_data

data_path_train = "/abs/path/to/tuh/edf/train"
data_path_eval  = "/abs/path/to/tuh/edf/eval"
save_path       = "/abs/path/to/tuh-eeg-ab-clean"  # will contain train/ and eval/ .npz

load_data(data_path_train, data_path_eval, save_path, sfreq=128, epoch_len_s=7.0)
```

This produces `.npz` epoch datasets for train/eval. The main pipeline (`main.py`) currently expects `.fif` files under `paths.data_path/{train,eval}`; use the preprocessing utility if you need cleaned epochs or to standardise raw EDF data ahead of `.fif` conversion.

---

## Data expectations
The current pipeline expects a TUH-style directory with preconverted `.fif` files and a simple split:
```
<data_path>/
├── train/
│   ├── abnormal/  # .fif files
│   └── normal/    # .fif files
└── eval/
    ├── abnormal/
    └── normal/
```
Each `.fif` should have `raw.info['subject_info']['sex']` as 1 or 2. Abnormal/normal is inferred from the folder name. Age is currently a placeholder (0) in the TUH path.

If starting from raw TUH EDFs, see `utils/cleanup_real_eeg_tuh.py` for a comprehensive cleaning pipeline and epoch export (produces `.npz`). That utility is separate from the `.fif` loader used by `main.py`.

---

## Configuration
Minimal current schema (example `config.yaml`):
```yaml
# Choose one of the supported methods (see Extraction methods)
method: wong_wang_avg

# Dataset corpus flag (kept for consistency; current path is TUH-centric)
data_corp: tuh

paths:
  # Root containing train/ and eval/ subfolders of .fif files
  data_path: "/absolute/path/to/tuh-eeg-ab-clean"
  # Where to write run artifacts
  results_root: "Results"

# Optuna/loader knobs (used by main.py)
optuna:
  n_trials: 10      # per task
  val_split: 0.1    # fixed split used globally
  patience: 7
  batch_size: 64

# Whether to ignore cached JSONL latent files and recompute
# (can also use CLI flag --reset)
# reset: false
```

CLI overrides:
```bash
python main.py --config config.yaml                 # normal run
python main.py --config config.yaml --reset         # force re-extraction
python main.py --config config.yaml --method c22    # override YAML method
```

Legacy configs under `configs/` include keys like `data_train`, `data_eval`, or `data_harvard`. The current loader reads `paths.data_path` and assumes `train/` and `eval/` inside it. Update older files accordingly before use.

---

## Running
Basic run:
```bash
python main.py --config config.yaml
```

Batch run across several methods (PBS example in this repo):
```bash
bash run_cleanup.sh     # enumerates many methods using --reset
bash run_all_configs.sh # if you update configs to the current schema
```

What happens during a run:
1. Load TUH `.fif` files from `paths.data_path/{train,eval}`.
2. Extract latent features for each file using the chosen `method`.
3. Cache JSONL latents to `Results/<data_corp>-<method>/temp_latent_features_{train,eval}.json`.
4. Run Optuna to pick simple head hyperparameters per task (classification heads for gender and abnormal; age head is currently skipped).
5. Evaluate and write reports/plots.

Caching: If the cache files exist and `--reset` is not passed, cached latents are reused. If counts mismatch the dataset size, latents are regenerated automatically.

---

## Outputs
Per run directory: `Results/<data_corp>-<method>/`
```
├── temp_latent_features_train.json
├── temp_latent_features_eval.json
├── final_metrics.txt     # flat text with inline descriptions
├── final_metrics.md      # human-friendly report
├── final_metrics.json    # raw metrics
├── pca_explained_variance_curve.png
├── train/
│   ├── hsic_matrix.png
│   ├── variance_hist.png
│   ├── pca2_scatter.png
│   └── tsne_scatter.png
├── eval/
│   ├── hsic_matrix.png
│   ├── variance_hist.png
│   ├── pca2_scatter.png
│   └── tsne_scatter.png
└── plots_<task>/         # per-task evaluation plots (if produced)
```

Metrics included (subset):
- Latent quality: active units, HSIC global score, KMeans cluster scores, PCA explained variance, geometry preservation.
- Dataset stats: sample counts and simple label distributions.
- Per-task head metrics for classification tasks (gender, abnormal).

---

## Extraction methods
Method names accepted by `--method` and `config.yaml`:
- Mechanistic
  - `ctm_cma_avg`, `ctm_cma_pc`: Cortico–Thalamic Model fitted with CMA-ES (average vs per-channel PSD).
  - `ctm_nn_avg`, `ctm_nn_pc`: CTM parameters via amortised regressor.
  - `jr_avg`, `jr_pc`: Jansen–Rit model fits.
  - `wong_wang_avg`, `wong_wang_pc`: Wong–Wang model fits.
  - `hopf_avg`, `hopf_pc`: Hopf oscillator model fits.
- Statistical
  - `c22`: Catch22 feature vector.
  - `pca_avg`, `pca_pc`: PCA features over PSD (frozen model files under `latent_extraction/pca/models/`).
- Learned
  - `psd_ae_avg`, `psd_ae_pc`: PSD autoencoder latents.
  - `eegnet`: EEGNet-based autoencoder.
  - `eeg2rep`: EEG2Rep representation extractor (requires `EEG2REP_CKPT` env var).

Notes:
- Some methods rely on local model files (e.g., `ctm_nn` regressor, PCA/PSD-AE checkpoints) already included under `latent_extraction/`.
- `eeg2rep` and `eegnet` may require GPU and additional assets; make sure dependencies are installed.

---

## HPC usage
Two example job scripts are provided; update paths/modules to your cluster:

- `run_all_configs.sh` (PBS): activates `~/env_thesis`, loads CUDA, and rebuilds `pycatch22` from source for compatibility:
  ```bash
  pip uninstall -y pycatch22
  pip install --no-cache-dir --no-binary=:all: pycatch22
  ```
  Then iterates configs with `python main.py --config ...`.

- `run_cleanup.sh` (PBS): calls `python main.py --reset --method <name>` across many methods.

- `run_latent_extraction.sh` (SLURM example): uses a different environment path; treat it as a template and adjust `cd` and `source` lines.

Tip: If `pycatch22` wheels are unavailable on your node, compile from source as shown above.

---

## Troubleshooting
- Missing/invalid labels: The TUH `.fif` loader expects `raw.info['subject_info']['sex']` ∈ {1,2}. Files outside this convention will emit warnings.
- Cache mismatch: If you move or change the dataset, pass `--reset` to recompute latents.
- CUDA OOM: Reduce `optuna.batch_size` in `config.yaml`.
- Slow CMA-ES fits: Use the amortised `ctm_nn_*` methods or statistical/learned methods for faster iteration.
- Legacy configs: Update `paths` to use `data_path` with `train/` and `eval/` inside. Remove unused keys like `data_harvard`.

---

## Dependencies
See `requirements.txt`. Notable:
- `torch`, `scikit-learn`, `optuna`, `mne`, `mne-bids`
- `pycatch22` (may require source build on HPC)
- `cma` (for CMA-ES model fits)
- `matplotlib`, `seaborn`

---

## Citation
If you use this code, please cite the thesis/work corresponding to this repository.