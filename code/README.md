## EEG Latent-Feature Pipeline

End-to-end pipeline for EEG latent-feature extraction and downstream evaluation/classification. Supports multiple datasets (TUH, LEMON) in a single run via a YAML config. Cached latent features are reused between runs unless reset.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Preprocessing](#preprocessing)
5. [Data Expectations](#data-expectations)
6. [Configuration](#configuration)
7. [Running](#running)
8. [Outputs](#outputs)
9. [Extraction Methods](#extraction-methods)
10. [HPC Usage](#hpc-usage)
11. [Troubleshooting](#troubleshooting)

---

## Features
- **Modular pipeline**: data loading → latent extraction → 5-fold cross-validation → evaluation → results.
- **Multi-dataset**: runs the full pipeline for every corpus listed in the config in a single invocation.
- **Many extraction options**: mechanistic models (CTM-CMA, CTM-NN, Jansen–Rit, Wong–Wang, Hopf), statistical (Catch22, PCA), and learned (EEGNet-AE, PSD-AE).
- **Config/CLI driven**: choose datasets, method, and optimisation knobs via YAML/flags.
- **Caching**: latent features written as JSONL and reused on subsequent runs.
- **Parallel processing**: CPU-based methods support multi-core extraction via `ProcessPoolExecutor`.
- **Subject-wise splitting**: subject-level `GroupKFold` / `GroupShuffleSplit` splits to prevent data leakage.
- **Comprehensive evaluation**: unsupervised latent metrics (HSIC independence, clustering, geometry) + supervised tasks (abnormality classification for TUH, age classification for LEMON) with both MLP and linear probe baselines (logistic regression for classification, Ridge for regression).
- **Reproducible results**: metrics and figures per run under `Results/`.

---

## Project Structure
```
code/
├── data_preprocessing/          # Data loading, cleaning (TUH, LEMON, Harvard)
│   ├── data_loading.py          # Loads train/eval pickle files
│   ├── cleanup_real_eeg_tuh.py  # TUH EDF → cleaned epochs pipeline
│   ├── cleanup_lemon.py         # LEMON dataset preprocessing
│   ├── cleanup_harvard.py       # Harvard dataset preprocessing
│   ├── cache_loading.py         # JSONL cache → DataLoader conversion
│   ├── gen_dataset.py           # Dataset class for (raw, g, a, ab, sample_id) tuples
│   └── harvard_python/          # Harvard download and cleaning scripts
├── latent_extraction/           # All extractors and pre-trained models
│   ├── extractor.py             # Unified extraction dispatcher
│   ├── cortico_thalamic.py      # CTM with CMA-ES fitting
│   ├── ctm_nn/                  # CTM via pre-trained neural network regressor
│   ├── jansen_rit.py            # Jansen–Rit neural mass model
│   ├── wong_wang.py             # Wong–Wang mean-field model
│   ├── hopf.py                  # Hopf (Stuart–Landau) oscillator model
│   ├── c22.py                   # Catch22 time-series features
│   ├── pca/                     # Frozen PCA models and extraction
│   ├── psd_ae/                  # PSD autoencoder (pre-trained models)
│   └── EEGNet_AE/               # EEGNet-based autoencoder
├── evaluation/                  # Latent metrics, cross-validation, model training
│   ├── metrix.py                # Unsupervised latent evaluation (HSIC, clustering, PCA, Shepard)
│   ├── cross_validation.py      # 5-fold subject-wise CV with Optuna + linear probe
│   ├── evaluation.py            # Results serialisation (final_metrics.txt)
│   ├── reporting.py             # Markdown report generation
│   ├── metrics_and_plots.py     # Cross-method comparison (CKA, Procrustes, CCA)
│   ├── generate_publication.py  # Publication-ready figures and tables
│   └── model_training/          # Optuna search + single-task MLP head
│       ├── optuna_search.py
│       └── single_task_model.py
├── utils/                       # PSD helpers, channel list, JSONL serialisation
│   ├── util.py
│   └── channel_name_test.py     # Canonical 19-channel name resolution
├── Results/                     # Auto-generated per-run outputs
├── configs/                     # YAML configuration files
│   └── default.yaml
├── main.py                      # Entry point
├── job_script.sh                # PBS batch script (update paths before use)
├── requirements.txt
└── README.md
```

---

## Setup
```bash
# Create and activate a virtual environment
python -m venv ~/env_thesis && source ~/env_thesis/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** On some HPC clusters, `pycatch22` may need to be compiled from source (see [HPC Usage](#hpc-usage)).

---

## Preprocessing

### TUH Abnormal EEG Corpus

If starting from raw TUH EDF files, run the cleaning pipeline first. It performs: channel renaming, bad-channel interpolation, edge trimming, bandpass filtering, ICA artifact removal (EOG/ECG), average re-referencing, artifact annotation, canonical 19-channel ordering, epoching, AutoReject, and beta/alpha QC.

```python
from data_preprocessing.cleanup_real_eeg_tuh import load_data

load_data(
    data_path_train="/path/to/tuh/edf/train",
    data_path_eval="/path/to/tuh/edf/eval",
    save_path="/path/to/tuh-eeg-ab-clean",
    sfreq=128,
    epoch_len_s=10.0,
)
```

### LEMON Dataset

The LEMON preprocessing script (`data_preprocessing/cleanup_lemon.py`) handles directory traversal of the untarred data, maps Initial IDs to INDI IDs via a lookup table, integrates age/sex metadata, filters for eyes-closed (EC) recordings, and applies shared cleaning logic from the TUH pipeline followed by AutoReject QC.

### Harvard Dataset

Download and cleaning utilities are in `data_preprocessing/harvard_python/`. Harvard is supported in the code (`cleanup_harvard.py`) but is not part of the primary benchmark.

---

## Data Expectations
The pipeline expects preprocessed pickle files in each dataset directory:
```
<data_path>/
├── train_epochs.pkl
└── eval_epochs.pkl
```

Each pickle contains a list of 5-tuples: `(raw, gender, age, abnormal, sample_id)`

| Field       | Type       | Description                                                        |
|-------------|------------|--------------------------------------------------------------------|
| `raw`       | `mne.Raw`  | Cleaned EEG data (19 channels, standardised montage)               |
| `gender`    | `int`      | 0 = female, 1 = male                                              |
| `age`       | `int/float`| Actual age for LEMON; 0 (placeholder) for TUH                     |
| `abnormal`  | `int`      | 0 = normal, 1 = abnormal (TUH); not used for LEMON                |
| `sample_id` | `str`      | Unique epoch identifier (encodes subject ID for group splitting)   |

---

## Configuration
Configuration via `configs/default.yaml`:
```yaml
method: ctm_nn_avg

datasets:
  lemon: "/path/to/Datasets/lemon"
  tuh: "/path/to/Datasets/tuh-eeg-ab-clean"

paths:
  results_root: "Results"

optuna:
  n_trials: 50       # Optuna trials per fold
  val_split: 0.15    # Fraction of training data for validation
  patience: 10       # Early-stopping patience
  batch_size: 64     # Batch size (automatically doubled for TUH)
```

**CLI overrides:**
```bash
python main.py --config configs/default.yaml
python main.py --config configs/default.yaml --method c22
python main.py --config configs/default.yaml --reset   # force re-extraction
python main.py --method jr_avg                          # uses default config
```

---

## Running

### Pipeline Workflow
For each dataset in the config, the pipeline executes:

1. **Data Loading** — load `train_epochs.pkl` and `eval_epochs.pkl`
2. **Latent Extraction** — extract features using the specified method (cached as JSONL)
3. **Unsupervised Latent Evaluation** — HSIC independence, clustering scores, geometry metrics
4. **5-Fold Cross-Validation** — subject-wise `GroupKFold` CV with Optuna hyperparameter search inside each fold
5. **Retrain** — train final MLP on full training set using best architecture from CV
6. **Final Evaluation** — evaluate on held-out eval set (MLP + linear probe)
7. **Save Results** — write `final_metrics.txt` and plots to `Results/{dataset}-{method}/`

### Dataset-Specific Tasks

| Dataset | Task                          | Target Field | Details                                   |
|---------|-------------------------------|--------------|-------------------------------------------|
| TUH     | Abnormality classification    | `abnormal`   | Binary: normal (0) vs abnormal (1)        |
| LEMON   | Age classification            | `age`        | Binary: young (<45) vs old (>=45)         |

### Batch Execution (HPC)
```bash
qsub job_script.sh
```
Edit `job_script.sh` to select which methods to run.

---

## Outputs
Results are organised in `Results/{dataset}-{method}/`:

### Per-Method Outputs
```
Results/{dataset}-{method}/
├── temp_latent_features_train.json    # Cached training latent features (JSONL)
├── temp_latent_features_eval.json     # Cached evaluation latent features (JSONL)
├── latent_metrics.json                # Unsupervised latent evaluation metrics
├── latent_space_analysis.png          # Combined latent-space overview
├── final_metrics.txt                  # Human-readable metrics summary
├── train/                             # Training set visualisations
│   ├── hsic_matrix.png               # Feature independence heatmap (HSIC)
│   ├── variance_hist.png             # Latent feature variance distribution
│   ├── pca2_scatter.png              # 2D PCA projection
│   └── shepard_plot.png              # Geometry preservation (Shepard diagram)
├── eval/                              # Evaluation set visualisations (same as train/)
├── plots_abnormal/                    # TUH: abnormality classification results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
└── plots_age/                         # LEMON: age classification results
    └── (same structure as plots_abnormal/)
```

### Aggregated Outputs (under `Results/`)
```
Results/
├── metrics_and_plots/                 # Cross-method comparison (CKA, Procrustes, CCA)
│   ├── tuh_small_aggregated/
│   ├── tuh_medium_unrestricted/
│   ├── lemon_small_aggregated/
│   └── lemon_medium_unrestricted/
├── publication_figures/               # Publication-ready figures and tables
├── summary_table.md                   # Comparative summary across all methods
└── summary_table.tex                  # LaTeX version of the summary table
```

---

## Extraction Methods
Method names accepted by `--method` and `config.yaml`:

### Mechanistic Models (Computational Brain Models)
| Method              | Description                                              |
|---------------------|----------------------------------------------------------|
| `ctm_cma_avg`       | Cortico–Thalamic Model, CMA-ES fit to average PSD        |
| `ctm_cma_pc`        | CTM, CMA-ES fit per channel                              |
| `ctm_nn_avg`        | CTM via pre-trained neural network (average PSD)         |
| `ctm_nn_pc`         | CTM via pre-trained neural network (per channel)         |
| `jr_avg`, `jr_pc`   | Jansen–Rit neural mass model                             |
| `wong_wang_avg`, `wong_wang_pc` | Wong–Wang mean-field model                  |
| `hopf_avg`, `hopf_pc` | Hopf (Stuart–Landau) oscillator model                  |

### Statistical Methods
| Method              | Description                                              |
|---------------------|----------------------------------------------------------|
| `c22`               | Catch22 time-series feature extraction (22 features)     |
| `pca_avg`, `pca_pc` | PCA over power spectral density (frozen models)          |

### Learned Representations
| Method              | Description                                              |
|---------------------|----------------------------------------------------------|
| `psd_ae_avg`, `psd_ae_pc` | Power Spectral Density Autoencoder                |
| `eegnet`            | EEGNet-based autoencoder (raw EEG input)                 |

### Parallelisation
- **CPU parallel** (`ProcessPoolExecutor`): `ctm_cma_*`, `jr_*`, `wong_wang_*`, `hopf_*`, `c22`
- **GPU accelerated**: `ctm_nn_*`, `psd_ae_*`, `eegnet`

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

### Considerations
- **Memory**: large datasets may require 32–64 GB RAM for parallel processing
- **CPU cores**: parallelisable methods benefit from high core counts (`n_workers=64`)
- **GPU**: neural network methods (`ctm_nn_*`, `psd_ae_*`, `eegnet`) benefit from GPU acceleration
- **pycatch22**: may need compilation from source on some clusters:
  ```bash
  pip uninstall -y pycatch22
  pip install --no-cache-dir --no-binary=:all: pycatch22
  ```

---

## Troubleshooting

| Problem                  | Solution                                                              |
|--------------------------|-----------------------------------------------------------------------|
| Missing pickle files     | Ensure `train_epochs.pkl` and `eval_epochs.pkl` exist in `data_path` |
| Cache mismatch           | Use `--reset` to recompute latent features                           |
| Memory errors            | Reduce `optuna.batch_size` or use fewer parallel workers             |
| GPU OOM                  | Reduce batch size or use CPU-only methods                            |
| Slow extraction          | Use `ctm_nn_*` instead of `ctm_cma_*`; enable parallel processing   |
| Missing sample IDs       | Pipeline falls back to random epoch splits (warns at runtime)        |

---

## Dependencies
See `requirements.txt`. Key packages:

| Category             | Packages                                            |
|----------------------|-----------------------------------------------------|
| Core                 | `numpy`, `scipy`, `scikit-learn`                    |
| Deep learning        | `torch`                                             |
| EEG processing       | `mne`, `mne-bids`, `autoreject`                     |
| Optimisation         | `optuna`, `cma`                                     |
| Feature extraction   | `pycatch22`                                         |
| Visualisation        | `matplotlib`, `seaborn`                             |
| Utilities            | `PyYAML`, `tqdm`                                    |

GPU acceleration requires a CUDA-compatible PyTorch installation.
