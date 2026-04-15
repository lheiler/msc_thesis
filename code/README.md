## EEG Latent-Feature Pipeline

End-to-end pipeline for EEG latent-feature extraction and downstream evaluation/classification. Supports multiple datasets (TUH, LEMON, Harvard) in a single run via a YAML config. Cached latent features are reused between runs unless reset.

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
- **Modular pipeline**: data loading → latent extraction → 5-fold cross-validation → evaluation → results.
- **Multi-dataset**: runs the full pipeline for every corpus listed in the config in a single invocation.
- **Many extraction options**: mechanistic models (CTM-CMA, CTM-NN, JR, Wong–Wang, Hopf), statistical (Catch22, PCA), and learned (EEGNet-AE, PSD-AE).
- **Config/CLI driven**: choose datasets, method, and optimisation knobs via YAML/flags.
- **Caching**: latent features written as JSONL and reused on subsequent runs.
- **Parallel processing**: CPU-based methods support parallel extraction for faster processing.
- **Subject-wise splitting**: Proper subject-level GroupKFold splits to prevent data leakage.
- **Comprehensive evaluation**: Unsupervised latent metrics (clustering, geometry, independence) + supervised tasks (abnormality classification, age classification) with both MLP and linear-probe baselines.
- **Reproducible results**: metrics and figures per run under `Results/`.

---

## Project Structure
```
code/
├── data_preprocessing/      # Data loading, cleaning (TUH, LEMON, Harvard)
├── latent_extraction/       # All extractors and pre-trained models
├── evaluation/              # Latent metrics, cross-validation, model training
│   ├── model_training/      # Optuna search + single-task MLP head
├── utils/                   # PSD helpers, channel list, JSONL serialisation
├── Results/                 # Auto-generated per-run outputs
├── configs/                 # YAML configuration files
├── main.py                  # Entry point
├── job_script.sh            # PBS batch script (update paths before use)
├── requirements.txt
└── README.md
```

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
Current schema (`configs/default.yaml`):
```yaml
# Choose one of the supported methods (see Extraction methods section)
method: ctm_nn_avg

# Multiple datasets – the pipeline loops over each corpus
datasets:
  lemon: "/path/to/Datasets/lemon"
  tuh: "/path/to/Datasets/tuh-eeg-ab-clean"

paths:
  results_root: "Results"

# Hyperparameter optimisation (Optuna, used inside cross-validation)
optuna:
  n_trials: 50       # Trials per fold
  val_split: 0.15    # Fraction of training data reserved for validation
  patience: 10       # Early-stopping patience within each trial
  batch_size: 64     # Batch size (doubled automatically for TUH)
```

**CLI Usage:**
```bash
# Basic run with config file
python main.py --config configs/default.yaml

# Force re-extraction of latent features (ignore cache)
python main.py --config configs/default.yaml --reset

# Override method from command line
python main.py --config configs/default.yaml --method c22

# Run with default config
python main.py --method jr_avg
```

**Important Notes:**
- The pipeline expects preprocessed pickle files (`train_epochs.pkl`, `eval_epochs.pkl`) in each dataset directory
- Results are organised as `{results_root}/{dataset_name}-{method}/`
- Latent features are cached as JSONL files and reused unless `--reset` is specified
- Subject-wise GroupKFold splitting prevents data leakage

---

## Running

### Single Method Execution
```bash
# Run with specific method (uses configs/default.yaml)
python main.py --method wong_wang_avg

# Run with explicit config file
python main.py --config configs/default.yaml

# Force re-extraction (ignore cached latent features)
python main.py --method c22 --reset
```

### Batch Execution
Use the provided PBS script to run multiple methods sequentially on a cluster:

```bash
qsub job_script.sh
```

Edit `job_script.sh` to select which methods to run.

### Pipeline Workflow
For each dataset in the config, the pipeline executes:

1. **Data Loading**: Load preprocessed pickle files (`train_epochs.pkl`, `eval_epochs.pkl`)
2. **Latent Extraction**: Extract features using the specified method (with caching)
3. **Unsupervised Latent Evaluation**: Compute clustering, geometry, and independence metrics
4. **5-Fold Cross-Validation**: Subject-wise GroupKFold CV with Optuna inside each fold
5. **Retrain**: Train final MLP using the best architecture from CV on the full training set
6. **Final Evaluation**: Evaluate on the held-out eval set (MLP + linear probe baseline)
7. **Save Results**: Write `final_metrics.txt` and plots to `Results/{dataset}-{method}/`

### Performance Optimisation
- **Parallel Processing**: CPU-based methods support parallel extraction via `n_workers` parameter
- **GPU Acceleration**: Neural network methods automatically use GPU when available
- **Caching**: Latent features are cached as JSONL to avoid re-computation across runs; reused unless `--reset` is specified or dataset size changes

---

## Outputs
Results are organised in: `Results/{dataset}-{method}/`

### Core Files
```
├── temp_latent_features_train.json    # Cached training latent features (JSONL)
├── temp_latent_features_eval.json     # Cached evaluation latent features (JSONL)
├── latent_metrics.json                # Unsupervised latent evaluation metrics
└── final_metrics.txt                  # Human-readable metrics with inline descriptions
```

### Visualisation Outputs
```
├── latent_space_analysis.png          # Combined latent-space overview
├── train/                             # Training set visualisations
│   ├── hsic_matrix.png               # Feature independence heatmap
│   ├── variance_hist.png             # Latent feature variance distribution
│   ├── pca2_scatter.png              # 2D PCA projection
│   └── tsne_scatter.png              # t-SNE embedding
├── eval/                              # Evaluation set visualisations
│   └── (same as train/)
├── plots_abnormal/                    # Abnormal classification results (TUH)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.png
└── plots_gender/                      # Gender classification results (TUH)
    └── (same structure as abnormal/)
```

### Aggregated Outputs (under `Results/`)
```
├── metrics_and_plots/                 # Cross-method comparison matrices (CKA, Procrustes, CCA)
├── publication_figures/               # Publication-ready figures and tables
├── summary_table.md                   # Comparative summary across all methods
└── summary_table.tex                  # LaTeX version of the summary table
```

### Evaluation Metrics Summary
- **Latent Quality**: Active units, feature independence (HSIC), clustering scores, geometry preservation
- **Downstream Performance**: Classification accuracy, F1-scores, ROC-AUC for abnormality/age tasks
- **Linear Probe Baseline**: Logistic regression baseline for each downstream task
- **Cross-Validation**: 5-fold subject-wise CV results with per-fold metrics
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
A PBS job script (`job_script.sh`) is provided. Update paths and modules for your system:

```bash
#!/bin/bash
#PBS -N final_eval
#PBS -q v1_large24
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=64:mem=128gb

cd /path/to/thesis/code
source ~/env_thesis/bin/activate

python main.py --method jr_avg
python main.py --method hopf_pc
# ... add more methods as needed
```

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
See `requirements.txt` for the core list. Key dependencies:

### Core Libraries
- **`torch`**: PyTorch for neural network methods and tensor operations
- **`scikit-learn`**: Machine learning utilities, PCA, clustering
- **`optuna`**: Hyperparameter optimisation framework
- **`numpy`, `scipy`**: Numerical computing

### EEG Processing
- **`mne`, `mne-bids`**: EEG data loading and preprocessing
- **`braindecode`**: EEG-specific deep learning utilities
- **`autoreject`**: Automated epoch rejection during preprocessing

### Method-Specific
- **`pycatch22`**: Catch22 time-series features (may require source build)
- **`cma`**: CMA-ES optimisation for mechanistic model fitting
- **`torcheeg`**: Additional EEG processing utilities
- **`numba`** *(optional)*: JIT compilation for faster CBM fitting

### Visualisation & Utilities
- **`matplotlib`, `seaborn`**: Plotting and visualisation
- **`pandas`**: Data manipulation (used in preprocessing and publication scripts)
- **`PyYAML`**: Configuration file parsing
- **`tqdm`**: Progress bars

### Installation Notes
- Some clusters may require `pycatch22` compilation from source
- GPU acceleration requires CUDA-compatible PyTorch installation
- Virtual environment recommended: `python -m venv ~/env_thesis`

---

## Citation
If you use this code, please cite the corresponding thesis/publication.