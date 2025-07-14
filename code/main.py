import faulthandler
faulthandler.enable()

from data_preprocessing import data_loading as dl

from model_training.single_task_model import SingleTaskModel, train as train_single_task
from evaluation.single_task_evaluation import evaluate_single_task

import evaluation.evaluation as eval  # keep generic helpers (e.g. HSIC + reporting)
import latent_extraction.extractor as extractor
import numpy as np
import os
import ast
from torch.utils.data import DataLoader, TensorDataset
import re
import json
import torch
import argparse
import yaml
from utils.latent_loading import (
    load_latent_parameters_array,
    load_latent_c22_parameters_array,
    load_latent_ae_parameters_array,)

# Dataset stats helper
from utils.data_metrics import compute_dataset_stats

def main():
    """
    Main function to run the entire pipeline.
    """
    # 0️⃣ Load configuration from YAML / CLI -------------------------------
    parser = argparse.ArgumentParser(description="EEG classification pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------  Parameters from config  ------------------------
    method    = cfg.get("method", "ctm")
    data_corp = cfg.get("data_corp", "harvard")

    paths_cfg = cfg.get("paths", {})
    data_path_train   = paths_cfg.get("data_train", "")
    data_path_eval    = paths_cfg.get("data_eval", "")
    data_path_harvard = paths_cfg.get("data_harvard", "")

    results_root = paths_cfg.get("results_root", "Results")
    # Derive an experiment‐specific identifier from the *actual* BIDS root that is being used. This
    # avoids different source folders writing into the same results directory 
    if data_corp == "harvard":
        dataset_root = data_path_harvard
    else:
        # TUH or any other corpus: fall back to the training directory path (eval may be empty)
        dataset_root = data_path_train or data_path_eval

    dataset_id = os.path.basename(os.path.normpath(dataset_root)) or data_corp
    results_path = os.path.join(results_root, f"{dataset_id}-{method}")
    os.makedirs(results_path, exist_ok=True)
    
    print(f"Results will be saved to: {results_path}")

    model_cfg = cfg.get("model", {})
    batch_size    = model_cfg.get("batch_size", 16)
    num_epochs    = model_cfg.get("num_epochs", 20)
    dropout       = model_cfg.get("dropout", 0.2)
    weight_decay  = model_cfg.get("weight_decay", 0.0)
    scheduler     = model_cfg.get("scheduler", "none")
    
    # If ``reset`` is True the latent feature files will be (re-)generated even
    # when they already exist. Otherwise the pipeline attempts to reuse cached
    # latents whenever their sample count matches the raw dataset.
    reset = cfg.get("reset", False)
    
    # --------------------------------------------------------------------
    # 1. Load raw EEG data (required for counts + potential extraction)
    # --------------------------------------------------------------------
    print("Loading raw EEG data …")
    if data_corp == "harvard":
        t_data, e_data = dl.load_data_harvard(data_path_harvard)
    else:
        t_data = dl.load_data(data_path_train)
        e_data = dl.load_data(data_path_eval)

    n_train, n_eval = len(t_data), len(e_data)

    # --------------------------------------------------------------------
    # 2. Decide whether to reuse cached latent features
    # --------------------------------------------------------------------
    def _latent_loader(split: str):
        path_stem = os.path.join(results_path, f"temp_latent_features_{split}")
        return load_latent_ae_parameters_array(path_stem, batch_size=batch_size)

    use_cache = not reset
    if use_cache:
        cache_exists = (
            os.path.exists(os.path.join(results_path, "temp_latent_features_train.json")) and
            os.path.exists(os.path.join(results_path, "temp_latent_features_eval.json"))
        )
        if cache_exists:
            t_latent_features = _latent_loader("train")
            e_latent_features = _latent_loader("eval")
            if len(t_latent_features.dataset) != n_train or len(e_latent_features.dataset) != n_eval:
                print("⚠️  Cached latent features do not match dataset size – regenerating …")
                use_cache = False
        else:
            use_cache = False

    # --------------------------------------------------------------------
    # 3. If required, extract fresh latent representations
    # --------------------------------------------------------------------
    if not use_cache:
        print("Extracting latent features …")
        t_latent_features = extractor.extract_latent_features(
            t_data,
            batch_size=batch_size,
            save_path=os.path.join(results_path, "temp_latent_features_train.json"),
            method=method,
        )
        e_latent_features = extractor.extract_latent_features(
            e_data,
            batch_size=batch_size,
            save_path=os.path.join(results_path, "temp_latent_features_eval.json"),
            method=method,
        )
    else:
        print("Cached latent features loaded successfully.")

    # --------------------------------------------------------------------
    # Safety check: ensure we have data before proceeding
    # --------------------------------------------------------------------
    if len(t_latent_features.dataset) == 0 or len(e_latent_features.dataset) == 0:
        msg = (
            "❌ No training or evaluation samples were loaded. "
            "Please verify the dataset paths and that preprocessing succeeded."
        )
        print(msg)
        return
    
    # --------------------------------------------------------------------
    # t-SNE visualisation removed – entire module has been deprecated.
    # --------------------------------------------------------------------
    # ------------------------    # 2. Train **independent** models per task  ------------------------

    print("Detecting tasks from dataset …")
    sample0 = t_latent_features.dataset[0]
    latent_vec = sample0[0].detach().clone()  # convert once for dimensionality
    input_dim = latent_vec.numel()
    num_tasks = len(sample0) - 1
    print(f"📈 Found {num_tasks} prediction task(s) – training separate networks for each.")

    metrics_all = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for task_idx in range(num_tasks):
        # ---------------- Data preparation per task ----------------
        x_train, y_raw_train = [], []
        for sample in t_latent_features.dataset:
            x_train.append(sample[0].detach().clone().float())
            y_raw_train.append(float(sample[task_idx + 1]))

        # ------------- Determine task type (simple heuristic) -------------
        uniq_vals = torch.unique(torch.tensor(y_raw_train))
        if uniq_vals.numel() <= 10 and torch.all((uniq_vals == 0) | (uniq_vals == 1) | (uniq_vals == 2)):
            task_type = "classification"
        else:
            task_type = "regression"

        # ----------- Infer human-readable task name ----------------------
        if task_type == "regression":
            task_name = "age"
        else:  # classification
            label_set = set(uniq_vals.tolist())
            # Abnormality annotations are 0/1 only. Gender uses 1/2 (optionally 0=unknown).
            if label_set.issubset({0.0, 1.0}):
                task_name = "abnormal"
            else:
                task_name = "gender"

        print(
            f"🔹 Task {task_idx+1}: detected as {task_type} → '{task_name}' "
            f"(unique values: {uniq_vals.tolist()})"
        )

        # 🗂  Finalise tensors & loaders ----------------------------------
        X_train = torch.stack(x_train)
        y_train_tensor = torch.tensor(y_raw_train, dtype=torch.float32)

        # For classification tasks with labels ∈ {1,2}, map → {0,1}
        if task_type == "classification":
            if torch.all((y_train_tensor == 1) | (y_train_tensor == 2)):
                y_train_tensor = (y_train_tensor == 2).float()

        train_loader = DataLoader(TensorDataset(X_train, y_train_tensor), batch_size=batch_size, shuffle=True)

        # Eval split ------------------------------------------------------
        x_eval, y_raw_eval = [], []
        for sample in e_latent_features.dataset:
            x_eval.append(sample[0].detach().clone().float())
            y_raw_eval.append(float(sample[task_idx + 1]))
        X_eval = torch.stack(x_eval)
        y_eval_tensor = torch.tensor(y_raw_eval, dtype=torch.float32)
        if task_type == "classification":
            if torch.all((y_eval_tensor == 1) | (y_eval_tensor == 2)):
                y_eval_tensor = (y_eval_tensor == 2).float()

        eval_loader = DataLoader(TensorDataset(X_eval, y_eval_tensor), batch_size=batch_size, shuffle=False)

        # ---------------- Model + training -----------------------
        model = SingleTaskModel(input_dim=input_dim, output_type=task_type, dropout=dropout)
        print(f"   → Training independent {task_type} network …")
        train_single_task(
            model,
            train_loader,
            n_epochs=num_epochs,
            weight_decay=weight_decay,
            scheduler=scheduler,
            device=device,
        )

        # ---------------- Evaluation -----------------------------
        task_metrics = evaluate_single_task(model, eval_loader, output_type=task_type, device=device)
        metrics_all[task_name] = task_metrics

        # Persist model weights
        #torch.save(model.state_dict(), os.path.join(results_path, f"task_{task_idx}_model.pth"))

    # ------------------------------------------------------------
    # 3. Dataset descriptive statistics & latent-space independence
    # ------------------------------------------------------------
    train_stats = compute_dataset_stats(t_latent_features)
    eval_stats  = compute_dataset_stats(e_latent_features)

    xs_eval = torch.stack([sample[0].detach().clone().float() for sample in e_latent_features.dataset])
    independence_scores = eval.independence_of_features(xs_eval, save_path=results_path)

    # 4. Collate and save final results --------------------------
    final_results = {
        "metrics_per_task": metrics_all,
        "train_dataset_stats": train_stats,
        "eval_dataset_stats": eval_stats,
        "global_independence_score": independence_scores["global_score"],
    }

    print("Saving results …")
    eval.save_results(final_results, results_path)

    print("✅ Pipeline completed successfully!")
    

    
if __name__ == "__main__":
    main()
