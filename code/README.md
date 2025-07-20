# EEG Latent-Feature Classification Pipeline

A research-grade, end-to-end pipeline for **EEG latent-feature extraction** and **multi-task classification** (gender, age, abnormality).  It is designed to be reproducible, configurable via YAML, and easy to extend.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Results & Outputs](#results--outputs)

---

## Features
* **Modular pipeline** – data loading, latent-feature extraction, model training, evaluation, and visualisation live in dedicated packages.
* **Multiple extraction methods** – `ctm`, `c22`, `c22_psd`, and deep-learning `AE`.
* **Independent per-task models** – trains *separate* lightweight MLPs for gender, age, and abnormality, avoiding parameter sharing.
* **YAML-driven config** – switch datasets, methods, hyper-parameters without touching code.
* **Snake-case package layout** – PEP-8 compliant, importable with `pip install -e .`.
* **Reproducible results** – each run writes to an auto-generated `Results/…` directory.

---

## Project Structure
```
code/
├── data_preprocessing/      # Raw EEG → cleaned tensors
├── latent_extraction/       # CTM, Catch22, Auto-encoder, …
├── model_training/          # ClassificationModel & trainer
├── evaluation/              # Metrics, HSIC independence, saving
├── Results/                 # <-- auto-generated metrics
├── configs/                 # YAML experiment files (optional)
├── main.py                  # Single entry-point
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
3. Train the multi-task classifier.
4. Evaluate and save metrics + figures.

Set `reset: true` in the YAML if you **want to force re-extraction** of latent features even if matching cached files already exist.  Leave it `false` (default) to reuse cached latents for faster iteration.

---

## Configuration
All parameters live in a YAML file.  Example (`config.yaml`):
```yaml
method: ctm               # ctm | c22 | c22_psd | AE
# Data corpus: harvard (BIDS) or tuh (TUH EEG)
data_corp: harvard

paths:
  data_train:   "/path/to/tuh/train"   # TUH only
  data_eval:    "/path/to/tuh/eval"    # TUH only
  data_harvard: "/path/to/bids/root"   # Harvard only
  results_root: "Results"              # where outputs go

model:
  batch_size: 16
  num_epochs: 20

extracted: false            # set true to reuse cached latents
```
Create as many configs as you like under `configs/`, e.g. `configs/ablation_c22.yaml`, and launch them with:
```bash
python main.py --config configs/ablation_c22.yaml
```

---

## Running the Pipeline
| Stage | Script / Module | Key Function |
|-------|-----------------|--------------|
| Data loading | `data_preprocessing/data_loading.py` | `load_data()` / `load_data_harvard()` |
| Latent extraction | `latent_extraction/extractor.py` | `extract_latent_features()` |
| Model training | `model_training/classification_model.py` | `train()` |
| Evaluation | `evaluation/evaluation.py` | `run_evaluation()` |
| HSIC independence | `evaluation/evaluation.py` | `independence_of_features()` |

Each stage can also be called independently in an interactive notebook for debugging.

---

## Results & Outputs
After a successful run you’ll find a folder like:
```
Results/bids_100_normal_abnormal_clean-ctm/
├── temp_latent_features_train.json
├── temp_latent_features_eval.json
├── final_metrics.txt
├── hsic_matrix.png
└── hsic_matrix.png          (pair-wise latent dependency)

In addition to classic metrics (loss, accuracy, MAE/RMSE), the **final_metrics.md** report now includes a *Model prediction distribution* table for each **classification** task, showing how often the network predicted each label (counts + %). 