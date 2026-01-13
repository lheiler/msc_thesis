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
- **Modular pipeline**: data loading → latent extraction → Optuna hyperparameter search → evaluation → reports.
- **Many extraction options**: mechanistic models (CTM-CMA, CTM-NN, JR, Wong–Wang, Hopf), statistical (Catch22, PCA), and learned (EEGNet-AE, PSD-AE).
- **Config/CLI driven**: choose dataset root, method, and optimisation knobs via YAML/flags.
- **Caching**: latent features written as JSONL and reused on subsequent runs.
- **Parallel processing**: CPU-based methods support parallel extraction for faster processing.
- **Subject-wise splitting**: Proper subject-level train/validation splits to prevent data leakage.
- **Comprehensive evaluation**: Unsupervised metrics (clustering, geometry) + supervised tasks (abnormal classification, gender classification).
- **Reproducible reports**: text, markdown, JSON, and figures per run under `Results/`.

---

## Project Structure
```
code/
├── data_preprocessing/      # TUH .fif loader
├── latent_extraction/       # All extractors and models
├── evaluation/              # Latent metrics, reporting, model training
│   ├── model_training/      # Optuna search + single-task head
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

This produces cleaned epoch data that can be saved as pickle files for the main pipeline. The preprocessing utility creates standardized epochs with consistent channelization, artifact removal, and quality control.

---

## Data expectations
The current pipeline expects a TUH-style directory with preprocessed pickle files:
```
<data_path>/
├── train_epochs.pkl  # List of tuples: (raw, gender, age, abnormal, sample_id)
└── eval_epochs.pkl   # List of tuples: (raw, gender, age, abnormal, sample_id)
```

**Data format**: Each tuple contains:
- `raw`: MNE Raw object with standardized EEG data
- `gender`: 0=female, 1=male
- `age`: Always 0 (placeholder for compatibility)
- `abnormal`: 0=normal, 1=abnormal
- `sample_id`: Unique epoch identifier

If starting from raw TUH EDFs, see `utils/cleanup_real_eeg_tuh.py` for a comprehensive cleaning pipeline and epoch export.

---

## Configuration
Current schema (example `config.yaml`):
```yaml
# Choose one of the supported methods (see Extraction methods section)
# Available: ctm_cma_avg, ctm_cma_pc, ctm_nn_avg, ctm_nn_pc, jr_avg, jr_pc,
#           wong_wang_avg, wong_wang_pc, hopf_avg, hopf_pc, c22, 
#           pca_avg, pca_pc, psd_ae_avg, psd_ae_pc, eegnet
method: wong_wang_avg

# Dataset corpus identifier (used for result directory naming)
data_corp: tuh

paths:
  # Root directory containing train_epochs.pkl and eval_epochs.pkl
  data_path: "~/thesis/Datasets/tuh-eeg-ab-clean"
  # Directory where all results will be written  
  results_root: "Results"

# Hyperparameter optimization settings
optuna:
  n_trials: 50      # Number of optimization trials per task
  val_split: 0.15   # Fraction of training data for validation (subject-wise split)
  patience: 7       # Early stopping patience for each trial
  batch_size: 512   # Batch size for data loading and training
```

**CLI Usage:**
```bash
# Basic run with config file
python main.py --config config.yaml

# Force re-extraction of latent features (ignore cache)
python main.py --config config.yaml --reset

# Override method from command line
python main.py --config config.yaml --method c22

# Run with default config.yaml
python main.py --method jr_avg
```

**Important Notes:**
- The pipeline expects preprocessed pickle files (`train_epochs.pkl`, `eval_epochs.pkl`) in the data directory
- Results are organized as `{results_root}/{data_corp}-{method}/`
- Latent features are cached as JSONL files and reused unless `--reset` is specified
- Subject-wise train/validation splitting prevents data leakage

---

## Running

### Single Method Execution
```bash
# Run with specific method
python main.py --method wong_wang_avg

# Run with config file
python main.py --config config.yaml

# Force re-extraction (ignore cached latent features)
python main.py --method c22 --reset
```

### Batch Execution
For running multiple methods, use the provided shell scripts:

```bash
# Example batch script (run.sh) - customize methods as needed
bash run.sh

# Force re-extraction for multiple methods
bash run_cleanup.sh
```

### Pipeline Workflow
1. **Data Loading**: Loads preprocessed pickle files from `data_path`
2. **Latent Extraction**: Extracts features using specified method (with caching)
3. **Hyperparameter Search**: Optuna optimization for downstream tasks
4. **Evaluation**: 
   - Unsupervised metrics (clustering, geometry, independence)
   - Supervised tasks (abnormal classification, gender classification)
5. **Results**: Saves metrics, plots, and reports to `Results/{data_corp}-{method}/`

### Performance Optimization
- **Parallel Processing**: CPU-based methods support parallel extraction via `n_workers` parameter
- **GPU Acceleration**: Neural network methods automatically use GPU when available
- **Caching**: Latent features are cached to avoid re-computation across runs

**Execution Details:**
1. **Data Loading**: Load preprocessed pickle files (`train_epochs.pkl`, `eval_epochs.pkl`)
2. **Latent Extraction**: Extract features using the specified method (with optional parallel processing)
3. **Caching**: Save latent features as JSONL files for reuse across runs
4. **Hyperparameter Search**: Optuna optimization for downstream task heads
5. **Evaluation**: Compute unsupervised and supervised metrics
6. **Reporting**: Generate plots, summaries, and structured results

**Caching Behavior**: Cached latent features are reused unless `--reset` is specified or dataset size changes.

---

## Outputs
Results are organized in: `Results/{data_corp}-{method}/`

### Core Files
```
├── temp_latent_features_train.json    # Cached training latent features
├── temp_latent_features_eval.json     # Cached evaluation latent features  
├── final_metrics.txt                  # Human-readable metrics with descriptions
├── final_metrics.md                   # Markdown summary report
├── final_metrics.json                 # Structured metrics in JSON format
└── study.db                          # Optuna hyperparameter search database
```

### Visualization Outputs
```
├── pca_explained_variance_curve.png   # PCA analysis of latent features
├── train/                            # Training set visualizations
│   ├── hsic_matrix.png              # Feature independence heatmap
│   ├── variance_hist.png            # Latent feature variance distribution
│   ├── pca2_scatter.png             # 2D PCA projection
│   └── tsne_scatter.png             # t-SNE embedding
├── eval/                             # Evaluation set visualizations
│   └── (same as train/)
├── plots_abnormal/                   # Abnormal classification results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.png
└── plots_gender/                     # Gender classification results
    └── (same structure as abnormal/)
```

### Evaluation Metrics Summary
- **Latent Quality**: Active units, feature independence (HSIC), clustering scores, geometry preservation
- **Downstream Performance**: Classification accuracy, F1-scores, ROC-AUC for abnormal/gender tasks
- **Dataset Statistics**: Sample counts, label distributions, train/eval splits

---

## Extraction methods
Method names accepted by `--method` and `config.yaml`:

### Mechanistic Models (Computational Brain Models)
- **`ctm_cma_avg`, `ctm_cma_pc`**: Cortico–Thalamic Model fitted with CMA-ES optimization
  - `avg`: Fit to average PSD across channels
  - `pc`: Fit separately per channel
- **`ctm_nn_avg`, `ctm_nn_pc`**: CTM parameters via pre-trained neural network regressor
  - Fast amortized inference alternative to CMA-ES fitting
- **`jr_avg`, `jr_pc`**: Jansen–Rit neural mass model fits
- **`wong_wang_avg`, `wong_wang_pc`**: Wong–Wang mean-field model fits  
- **`hopf_avg`, `hopf_pc`**: Hopf (Stuart-Landau) oscillator model fits

### Statistical Methods
- **`c22`**: Catch22 time-series feature extraction (22 canonical features)
- **`pca_avg`, `pca_pc`**: Principal Component Analysis over power spectral density
  - Uses frozen PCA models under `latent_extraction/pca/models/`

### Learned Representations (Deep Learning)
- **`psd_ae_avg`, `psd_ae_pc`**: Power Spectral Density Autoencoder
  - `avg`: Average features across channels
  - `pc`: Per-channel features (concatenated)
- **`eegnet`**: EEGNet-based autoencoder for raw EEG

### Performance Notes
- **Parallel processing**: Methods marked with ⚡ support multi-core processing when `n_workers > 1`
  - ⚡ `ctm_cma_pc`, `ctm_cma_avg`, `jr_pc`, `jr_avg`, `wong_wang_pc`, `wong_wang_avg`, `hopf_pc`, `hopf_avg`, `c22`
- **GPU acceleration**: `ctm_nn_*`, `psd_ae_*`, `eegnet` methods benefit from GPU when available
- **Model dependencies**: Some methods require pre-trained models included in the repository

---

## HPC Usage
Example job scripts are provided for cluster environments. Update paths and modules for your system:

### PBS Scripts
- **`run.sh`**: Basic PBS script for running specific methods
  ```bash
  #!/bin/bash
  #PBS -lwalltime=24:00:00
  #PBS -q v1_large24
  #PBS -lselect=1:ncpus=64:mem=64gb
  
  cd /path/to/thesis/code
  source ~/env_thesis/bin/activate
  
  python main.py --method jr_pc
  python main.py --method hopf_pc
  ```

- **`run_cleanup.sh`**: Force re-extraction across multiple methods using `--reset`

### Key HPC Considerations
- **Memory**: Large datasets may require 32-64GB RAM for parallel processing
- **CPU cores**: Parallelizable methods benefit from high core counts (set `n_workers=64`)
- **GPU**: Neural network methods (`ctm_nn_*`, `psd_ae_*`, `eegnet`) benefit from GPU acceleration
- **pycatch22**: May need compilation from source on some clusters:
  ```bash
  pip uninstall -y pycatch22
  pip install --no-cache-dir --no-binary=:all: pycatch22
  ```

---

## Troubleshooting

### Common Issues
- **Missing pickle files**: Ensure `train_epochs.pkl` and `eval_epochs.pkl` exist in `data_path`
- **Cache mismatch**: If dataset changes, use `--reset` to recompute latent features
- **Memory errors**: 
  - Reduce `optuna.batch_size` in config
  - Use fewer parallel workers for CPU methods
  - Ensure sufficient RAM for large datasets
- **GPU issues**:
  - CUDA OOM: Reduce batch size or use CPU-only methods
  - Missing GPU: Pipeline automatically falls back to CPU
- **Slow extraction**: 
  - Use `ctm_nn_*` instead of `ctm_cma_*` for faster CTM fitting
  - Enable parallel processing for supported methods
  - Consider statistical methods (`c22`, `pca_*`) for quick iteration

### Data Format Issues
- **Invalid gender labels**: Pipeline expects 0=female, 1=male
- **Missing sample IDs**: May fall back to per-epoch rather than subject-wise splits
- **Age placeholder**: Age is currently set to 0 for all samples (compatibility)

### Dependencies
- **pycatch22**: May require source compilation on some systems
- **MNE**: Ensure compatible version for EEG data loading
- **CUDA**: Optional but recommended for neural network methods

---

## Dependencies
See `requirements.txt` for complete list. Key dependencies:

### Core Libraries
- **`torch`**: PyTorch for neural network methods and tensor operations
- **`scikit-learn`**: Machine learning utilities, PCA, clustering
- **`optuna`**: Hyperparameter optimization framework
- **`numpy`, `scipy`**: Numerical computing

### EEG Processing
- **`mne`, `mne-bids`**: EEG data loading and preprocessing
- **`braindecode==0.7.1`**: EEG-specific deep learning utilities

### Method-Specific
- **`pycatch22`**: Catch22 time-series features (may require source build)
- **`cma`**: CMA-ES optimization for mechanistic model fitting
- **`torcheeg`**: Additional EEG processing utilities

### Visualization & Utilities
- **`matplotlib`, `seaborn`**: Plotting and visualization
- **`PyYAML`**: Configuration file parsing
- **`tqdm`**: Progress bars

### Installation Notes
- Some clusters may require `pycatch22` compilation from source
- GPU acceleration requires CUDA-compatible PyTorch installation
- Virtual environment recommended: `python -m venv ~/env_thesis`

---

## Citation
If you use this code, please cite the corresponding thesis/publication.