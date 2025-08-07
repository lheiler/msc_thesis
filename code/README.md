# EEG Latent-Feature Classification Pipeline

A research-grade, end-to-end pipeline for **EEG latent-feature extraction** and **multi-task classification** (gender, age, abnormality). It is designed to be reproducible, configurable via YAML, and easy to extend.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Results & Outputs](#results--outputs)
7. [Advanced Features](#advanced-features)

---

## Features
* **Modular pipeline** – data loading, latent-feature extraction, model training, evaluation, and visualisation live in dedicated packages.
* **Multiple extraction methods** – `ctm`, `c22`, `c22_psd`, and deep-learning `AE` (Auto-encoder).
* **Independent per-task models** – trains *separate* lightweight MLPs for gender, age, and abnormality, avoiding parameter sharing.
* **YAML-driven config** – switch datasets, methods, hyper-parameters without touching code.
* **Snake-case package layout** – PEP-8 compliant, importable with `pip install -e .`.
* **Reproducible results** – each run writes to an auto-generated `Results/…` directory.
* **Hyperparameter optimization** – optional Optuna integration for automatic hyperparameter tuning.
* **Multi-dataset support** – Harvard BIDS format and TUH EEG corpus.

---

## Project Structure
```
code/
├── data_preprocessing/      # Raw EEG → cleaned tensors
├── latent_extraction/       # CTM, Catch22, Auto-encoder, CWT-AE
├── model_training/          # SingleTaskModel & Optuna trainer
├── evaluation/              # Metrics, HSIC independence, saving
├── utils/                   # Data loading, reporting utilities
├── Results/                 # <-- auto-generated metrics
├── configs/                 # YAML experiment files
├── extras/                  # Additional models and archives
├── main.py                  # Single entry-point
├── run_all_configs.sh      # Batch execution script
├── run_latent_extraction.sh # Latent extraction only
├── run_cleanup.sh          # Cleanup utilities
├── requirements.txt         # Python deps (>= versions)
└── README.md                
```
*Large datasets are **not** stored in the repo – supply paths in `config.yaml` or your own YAML file.*

---

## Quick Start
```bash
# 1. Clone & create an isolated Python environment (optional)
git clone <your-fork-url> eeg-pipeline && cd eeg-pipeline
python -m venv .venv && source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Edit config.yaml to point to your dataset locations (see below)

# 4. Run the pipeline
python main.py --config config.yaml
```

The first run will:
1. Load the dataset(s) defined in your config.
2. Extract latent features and cache them as JSON files in `Results/<corpus>-<method>-parameters*/`.
3. Train the multi-task classifier (with optional Optuna optimization).
4. Evaluate and save metrics + figures.

Set `reset: true` in the YAML if you **want to force re-extraction** of latent features even if matching cached files already exist. Leave it `false` (default) to reuse cached latents for faster iteration.

---

## Configuration
All parameters live in a YAML file. Example (`config.yaml`):
```yaml
method: c22               # ctm | c22 | c22_psd | AE
# Data corpus: harvard (BIDS) or tuh (TUH EEG)
data_corp: tuh

paths:
  data_tuh: "/path/to/tuh/eeg/data"      # TUH corpus path
  data_harvard: "/path/to/bids/root"     # Harvard BIDS path
  results_root: "Results"                 # where outputs go

# Manual hyperparameters (ignored if Optuna enabled)
model:
  batch_size: 16
  num_epochs: 30
  hidden_dims: [64]      # Single int or list
  dropout: 0.1
  weight_decay: 0.0
  scheduler: "none"       # none | cosine | step

# Hyper-parameter optimisation via Optuna (overrides manual settings)
optuna:
  n_trials: 10           # Number of optimisation trials per task
  val_split: 0.2         # Fraction of training data for validation
  patience: 5            # Early-stopping patience within each trial

# Whether to re-extract latent representations even when cached files exist
reset: false
```

Create as many configs as you like under `configs/`, e.g. `configs/ablation_c22.yaml`, and launch them with:
```bash
python main.py --config configs/ablation_c22.yaml
```

**Pre-configured experiments** are available in the `configs/` directory for different datasets and methods.

---

## Running the Pipeline

### Basic Execution
```bash
# Single experiment
python main.py --config config.yaml

# Multiple experiments
bash run_all_configs.sh
```

### Pipeline Stages
| Stage | Script / Module | Key Function |
|-------|-----------------|--------------|
| Data loading | `data_preprocessing/data_loading.py` | `load_data()` / `load_data_harvard()` |
| Latent extraction | `latent_extraction/extractor.py` | `extract_latent_features()` |
| Model training | `model_training/single_task_model.py` | `train_single_task()` |
| Hyperparameter tuning | `model_training/optuna_search.py` | `tune_hyperparameters()` |
| Evaluation | `evaluation/evaluation.py` | `run_evaluation()` |
| HSIC independence | `evaluation/evaluation.py` | `independence_of_features()` |

Each stage can also be called independently in an interactive notebook for debugging.

### Batch Execution
The `run_all_configs.sh` script runs multiple pre-configured experiments:
- Different datasets (500/2000 samples)
- Different extraction methods (CTM, C22, C22_PSD, AE)
- Different tasks (abnormality, age classification)

---

## Results & Outputs
After a successful run you'll find a folder like:
```
Results/bids_500_normal_abnormal_clean-c22/
├── temp_latent_features_train.json
├── temp_latent_features_eval.json
├── final_metrics.txt
├── final_metrics.md
├── hsic_matrix.png
├── task_0_model.pth          # Model weights (if saved)
└── optuna_trials.db          # Optuna database (if used)
```

### Metrics Included
- **Classification tasks**: Accuracy, Precision, Recall, F1-Score
- **Regression tasks**: MAE, RMSE, R²
- **Model prediction distribution** table for each classification task
- **HSIC independence scores** for latent feature analysis
- **Dataset statistics** for train/eval splits

---

## Advanced Features

### Hyperparameter Optimization
Enable Optuna for automatic hyperparameter tuning:
```yaml
optuna:
  n_trials: 30
  val_split: 0.2
  patience: 10
```

### Supported Extraction Methods
- **CTM** (`ctm`): Cortico-thalamic modeling
- **Catch22** (`c22`): 22 time-series features
- **Catch22 PSD** (`c22_psd`): Power spectral density features
- **Auto-encoder** (`AE`): Deep learning-based compression

### Dataset Support
- **Harvard BIDS**: Clean, structured EEG data
- **TUH EEG**: Temple University Hospital corpus

### Model Architecture
- **SingleTaskModel**: Independent MLP for each task
- **No parameter sharing** between tasks
- **Configurable architecture** via `hidden_dims`
- **Early stopping** and learning rate scheduling

### Utilities
- **Data metrics**: `utils/data_metrics.py`
- **Latent loading**: `utils/latent_loading.py`
- **Reporting**: `utils/reporting.py`
- **Harvard labels**: `utils/harvard_labels.py`

---

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size` in config
2. **Slow extraction**: Use `reset: false` to reuse cached latents
3. **Optuna trials**: Check `optuna_trials.db` for optimization history

### Performance Tips
- Use GPU acceleration when available
- Enable Optuna for better hyperparameters
- Reuse cached latent features for faster iteration
- Use pre-configured experiments in `configs/`

---

## Dependencies
Key dependencies (see `requirements.txt` for full list):
- `torch` - Deep learning framework
- `mne` - EEG processing
- `pycatch22` - Time-series features
- `optuna` - Hyperparameter optimization
- `torcheeg` - EEG-specific utilities
- `scikit-learn` - Machine learning utilities 