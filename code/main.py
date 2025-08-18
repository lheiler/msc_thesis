from data_preprocessing import data_loading as dl
from model_training.optuna_search import tune_hyperparameters
import evaluation.evaluation as eval
import latent_extraction.extractor as extractor
import numpy as np
import os
import ast
from torch.utils.data import DataLoader, TensorDataset, Subset
import re
import json
import torch
import argparse
import yaml
from utils.latent_loading import load_latent_parameters_array
from utils.data_metrics import compute_dataset_stats
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import evaluation.metrix as metrics
from pathlib import Path




def main():
    """
    Main function to run the entire pipeline.
    """

    # ------------------------------------------------------------------
    # 1) Parse CLI arguments and load configuration
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="EEG classification pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--reset", action="store_true", help="Reset the pipeline")
    parser.add_argument("--method", type=str, help="Method to use for latent feature extraction")
    args = parser.parse_args()
    reset = args.reset 

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 2) Core config and paths
    # ------------------------------------------------------------------
    method    = cfg.get("method")
    data_corp = cfg.get("data_corp")
    
    # If --method is supplied, it overrides config; otherwise keep config value
    method = args.method if args.method is not None else method

    paths_cfg = cfg.get("paths", {})
    data_path   = os.path.expanduser(paths_cfg.get("data_path", ""))
    data_path_train = os.path.join(data_path, "train")
    data_path_eval = os.path.join(data_path, "eval")
    results_root = paths_cfg.get("results_root", "Results")
    
    results_path = os.path.join(results_root, f"{data_corp}-{method}")
    os.makedirs(results_path, exist_ok=True)
    
    print(f"Results will be saved to: {results_path}")

    model_cfg   = cfg.get("model", {})

    # ------------------------------------------------------------------
    # 3) Optuna and training hyperparameters
    # ------------------------------------------------------------------
    optuna_cfg  = cfg.get("optuna", {})
    n_trials_opt   = optuna_cfg.get("n_trials", 30)
    val_split_opt  = optuna_cfg.get("val_split", 0.2)
    patience_opt   = optuna_cfg.get("patience", 10)
    batch_size   = optuna_cfg.get("batch_size", 64)
    
    n_train = sum(1 for _ in Path(data_path_train).rglob("*.fif"))
    n_eval  = sum(1 for _ in Path(data_path_eval).rglob("*.fif"))
    
    # ------------------------------------------------------------------
    # 4) Latent feature loading: cache or compute
    # ------------------------------------------------------------------
    def _latent_loader(split: str):
        return load_latent_parameters_array(
            os.path.join(results_path, f"temp_latent_features_{split}"),
            batch_size=batch_size,
        )


    train_cache = os.path.join(results_path, "temp_latent_features_train.json")
    eval_cache  = os.path.join(results_path, "temp_latent_features_eval.json")
    use_cache = (not reset and os.path.exists(train_cache) and os.path.exists(eval_cache))
    if use_cache:
        t_latent_features = _latent_loader("train")
        e_latent_features = _latent_loader("eval")
        if len(t_latent_features.dataset) != n_train or len(e_latent_features.dataset) != n_eval:
            print("⚠️  Cached latent features do not match dataset size – regenerating …")
            use_cache = False
        else: print("Cached latent features loaded successfully.")

    else:
        print("Loading raw EEG data …")
        t_data, e_data = dl.load_data(data_path_train), dl.load_data(data_path_eval)
        print(f"Loaded {len(t_data)} training samples and {len(e_data)} evaluation samples from tuh dataset")

        print("Extracting latent features …")
        t_latent_features = extractor.extract_latent_features(t_data, batch_size=batch_size, save_path=os.path.join(results_path, "temp_latent_features_train.json"), method=method)
        e_latent_features = extractor.extract_latent_features(e_data, batch_size=batch_size, save_path=os.path.join(results_path, "temp_latent_features_eval.json"), method=method)
    
        

    features_train = torch.stack([sample[0] for sample in t_latent_features.dataset])
    features_eval  = torch.stack([sample[0] for sample in e_latent_features.dataset])


    # ------------------------------------------------------------------
    # 5) Safety check: ensure expected sample counts
    # ------------------------------------------------------------------
    if len(t_latent_features.dataset) != n_train or len(e_latent_features.dataset) != n_eval:
        print("❌ Not enough training or evaluation samples were loaded.")
        return

    # ------------------------------------------------------------------
    # 6) Latent evaluation (disabled)
    # ------------------------------------------------------------------
    print("Evaluating latent features …")
    latent_metrics = metrics.evaluate_latent_features(t_latent_features, e_latent_features, results_path)

    # ------------------------------------------------------------------
    # 7) Training – separate models per task
    # ------------------------------------------------------------------
    print("Training models for each task")
    input_dim = t_latent_features.dataset[0][0].numel()
    num_tasks = len(t_latent_features.dataset[0]) - 1
    metrics_all = {}
    hyperparams_all = {}
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Fixed global train/val index split
    all_indices = list(range(len(t_latent_features.dataset)))
    val_split_frac = val_split_opt
    train_indices_global, val_indices_global = train_test_split(
        all_indices,
        test_size=val_split_frac,
        random_state=42,
        shuffle=True,
    )

    # Hard-coded tasks: 0) gender (clf), 1) age (regr), 2) abnormal (clf)
    task_map = {
        0: ("classification", "gender"),
        1: ("regression", "age"),
        2: ("classification", "abnormal"),
    }

    def build_xy(dataset, task_index):
        X = torch.stack([s[0].detach().clone().float() for s in dataset])
        y = torch.tensor([float(s[task_index + 1]) for s in dataset], dtype=torch.float32)
        return X, y

    def map_class_labels(y_tensor):
        return (y_tensor == 2).float() if torch.all((y_tensor == 1) | (y_tensor == 2)) else y_tensor

    for task_idx in range(num_tasks):
        if task_idx == 1:
            continue
        # Resolve task type/name and announce
        task_type, task_name = task_map.get(task_idx, ("classification", f"task_{task_idx+1}"))
        print(f"🔹 Task {task_idx+1}: hardcoded as {task_type} → '{task_name}'")

        # Build train tensors
        X_train, y_train_tensor = build_xy(t_latent_features.dataset, task_idx)
        if task_type == "classification":
            y_train_tensor = map_class_labels(y_train_tensor)
        assert X_train.shape[0] == y_train_tensor.shape[0], "Mismatch: features and labels have different lengths (train)."

        # Datasets and loaders (global split)
        train_dataset_full = TensorDataset(X_train, y_train_tensor)
        val_dataset_full   = TensorDataset(X_train.clone(), y_train_tensor.clone())
        train_loader = DataLoader(Subset(train_dataset_full, train_indices_global), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(train_dataset_full, val_indices_global),   batch_size=batch_size, shuffle=True)

        # Build eval tensors
        X_eval, y_eval_tensor = build_xy(e_latent_features.dataset, task_idx)
        if task_type == "classification":
            y_eval_tensor = map_class_labels(y_eval_tensor)
        assert X_eval.shape[0] == y_eval_tensor.shape[0], "Mismatch: features and labels have different lengths (eval)."
        eval_loader = DataLoader(TensorDataset(X_eval, y_eval_tensor), batch_size=batch_size, shuffle=True)

        print(f"   → Optuna search (n_trials={n_trials_opt}) for {task_type} …")
        search_out = tune_hyperparameters(
            train_loader,
            val_loader,
            input_dim=input_dim,
            output_type=task_type,
            n_trials=n_trials_opt,
            device=device,
            val_split=val_split_opt,
            early_stopping_patience=patience_opt,
            results_dir=results_path,
        )
        
        model = search_out["best_model"]
        best_params = search_out["best_params"]

        task_plot_dir = os.path.join(results_path, f"plots_{task_name}")
        os.makedirs(task_plot_dir, exist_ok=True)
        task_metrics = model.evaluate(
            eval_loader,
            output_type=task_type,
            device=device,
            plot_dir=task_plot_dir,
        )

        metrics_all[task_name] = task_metrics
        hyperparams_all[task_name] = best_params


    # ------------------------------------------------------------------
    # 8) Dataset statistics
    # ------------------------------------------------------------------
    train_stats = compute_dataset_stats(t_latent_features)
    eval_stats  = compute_dataset_stats(e_latent_features)

    # ------------------------------------------------------------------
    # 9) Collate and persist results
    # ------------------------------------------------------------------
    final_results = {
        "metrics_per_task": metrics_all,
        "hyperparams_per_task": hyperparams_all,
        "train_dataset_stats": train_stats,
        "eval_dataset_stats": eval_stats,
        "latent": latent_metrics,
    }

    print("Saving results …")
    eval.save_results(final_results, results_path)

    print(f"✅ Pipeline completed successfully to the path {results_path}!")
    

    
if __name__ == "__main__":
    main()
    
