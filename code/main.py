from data_preprocessing import data_loading as dl
from evaluation.model_training.optuna_search import tune_hyperparameters
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
from data_preprocessing.cache_loading import load_latent_parameters_array
from evaluation.data_metrics import compute_dataset_stats
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.decomposition import PCA
import evaluation.metrix as metrics
from pathlib import Path
import pickle


#todo: add timer for how long feature extraction takes


def main():
    """
    Main function to run the entire pipeline.
    """

    # ------------------------------------------------------------------
    # 1) Parse CLI arguments and load configuration
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="EEG classification pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration file")
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
    
    # Count epochs from pickle files instead of individual .fif files
    train_pickle = os.path.join(data_path, "train_epochs.pkl")
    eval_pickle = os.path.join(data_path, "eval_epochs.pkl")
    
    if os.path.exists(train_pickle) and os.path.exists(eval_pickle):
        with open(train_pickle, 'rb') as f:
            n_train = len(pickle.load(f))
        with open(eval_pickle, 'rb') as f:
            n_eval = len(pickle.load(f))
    else:
        print("⚠️  Pickle files not found. Run preprocessing first.")
        n_train, n_eval = 0, 0
    
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
        t_data, e_data = dl.load_data(data_path, "train"), dl.load_data(data_path, "eval")
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
    # 6) Latent evaluation
    # ------------------------------------------------------------------
    print("Evaluating latent features …")
    try:
        latent_metrics = metrics.evaluate_latent_features(t_latent_features, e_latent_features, results_path)
    except Exception as e:
        print(f"⚠️ Latent evaluation failed: {e}")
        latent_metrics = None
    
    
    # ------------------------------------------------------------------
    # 7) Training – separate models per task
    # ------------------------------------------------------------------
    print("Training models for each task")
    input_dim = t_latent_features.dataset[0][0].numel()
    num_tasks = len(t_latent_features.dataset[0]) - 1
    metrics_all = {}
    hyperparams_all = {}
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Fixed global train/val index split – SUBJECT-WISE to avoid leakage
    def _extract_subject_id(sample_id: str) -> str:
        # Try to extract base subject before session/trial markers
        # e.g., "aaaaapjb_s001_t001_epoch0000" → "aaaaapjb"
        m = re.match(r"^([A-Za-z0-9]+)_s\d+", sample_id)
        if m:
            return m.group(1)
        # Fallback: split at "_t" (trial) if present
        if "_t" in sample_id:
            return sample_id.split("_t", 1)[0].split("_s", 1)[0]
        # Fallback: split at first underscore
        if "_" in sample_id:
            return sample_id.split("_", 1)[0]
        return sample_id

    sample_ids_train = getattr(t_latent_features, "sample_ids", None)
    if sample_ids_train and len(sample_ids_train) == len(t_latent_features.dataset):
        subject_groups = [_extract_subject_id(sid) for sid in sample_ids_train]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_split_opt, random_state=42)
        split_iter = gss.split(list(range(len(subject_groups))), groups=subject_groups)
        train_indices_global, val_indices_global = next(split_iter)
        print(f"Using subject-wise split: {len(set([subject_groups[i] for i in train_indices_global]))} subjects train | "
              f"{len(set([subject_groups[i] for i in val_indices_global]))} subjects val")
    else:
        # Safe fallback to per-epoch random split if IDs missing
        all_indices = list(range(len(t_latent_features.dataset)))
        train_indices_global, val_indices_global = train_test_split(
            all_indices,
            test_size=val_split_opt,
            random_state=42,
            shuffle=True,
        )
        print("WARNING: sample_ids missing; falling back to per-epoch random split.")

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
        """
        Normalize gender labels to binary 0/1 with 1=male, 0=female.
        - If labels are {1,2} (legacy: 1=male, 2=female), map to (y==1).
        - If labels are {0,1} (current cleanup: 0=female, 1=male), leave as-is.
        - Otherwise, return unchanged (caller may handle unknown/missing).
        """
        if torch.all((y_tensor == 1) | (y_tensor == 2)):
            return (y_tensor == 1).float()
        if torch.all((y_tensor == 0) | (y_tensor == 1)):
            return y_tensor.float()
        return y_tensor

    def discretize_age(y_tensor):
        """
        Map continuous age to 5-year bins for LEMON:
        20-25 -> 0, 25-30 -> 1, ..., 75-80 -> 11
        """
        # Bins: [20, 25, 30, ..., 75, 80]
        bins = torch.arange(20, 81, 5).float()
        # Find index. Clamp to min/max bin index.
        # torch.bucketize finds boundaries; subtract 1 for 0-indexing
        indices = torch.bucketize(y_tensor, bins) - 1
        return torch.clamp(indices, 0, len(bins) - 2).float()

    for task_idx in range(num_tasks):
        # Resolve task type/name and announce
        task_type, task_name = task_map.get(task_idx, ("classification", f"task_{task_idx+1}"))
        
        # --- Task overrides for specific datasets ---
        num_classes = 1
        ordinal_sigma = None
        if data_corp == "lemon" and task_name == "age":
            task_type = "classification"
            num_classes = 12 # 5-year bins from 20 to 80
            ordinal_sigma = 1.0 # Standard smoothing for ordinal bins
            print(f"🔹 Task {task_idx+1}: [LEMON] Adjusting '{task_name}' to {task_type} with {num_classes} bins (ordinal_sigma={ordinal_sigma})")
        else:
            print(f"🔹 Task {task_idx+1}: hardcoded as {task_type} → '{task_name}'")

        # Build train tensors
        X_train, y_train_tensor = build_xy(t_latent_features.dataset, task_idx)
        if task_type == "classification":
            if data_corp == "lemon" and task_name == "age":
                 y_train_tensor = discretize_age(y_train_tensor)
            else:
                 y_train_tensor = map_class_labels(y_train_tensor)
        assert X_train.shape[0] == y_train_tensor.shape[0], "Mismatch: features and labels have different lengths (train)."
        
        # Normalize features using only the training split (avoid leakage)
        train_idx_tensor = torch.as_tensor(train_indices_global, dtype=torch.long)
        X_mean = X_train.index_select(0, train_idx_tensor).mean(dim=0, keepdim=True)
        X_std = X_train.index_select(0, train_idx_tensor).std(dim=0, keepdim=True) + 1e-8
        X_train = (X_train - X_mean) / X_std

        # Datasets and loaders (global split)
        train_dataset_full = TensorDataset(X_train, y_train_tensor)
        train_loader = DataLoader(Subset(train_dataset_full, train_indices_global), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(train_dataset_full, val_indices_global),   batch_size=batch_size, shuffle=True)

        # Build eval tensors
        X_eval, y_eval_tensor = build_xy(e_latent_features.dataset, task_idx)
        if task_type == "classification":
            if data_corp == "lemon" and task_name == "age":
                 y_eval_tensor = discretize_age(y_eval_tensor)
            else:
                 y_eval_tensor = map_class_labels(y_eval_tensor)
        assert X_eval.shape[0] == y_eval_tensor.shape[0], "Mismatch: features and labels have different lengths (eval)."
        
        # Apply same normalization as training data
        X_eval = (X_eval - X_mean) / X_std
        eval_loader = DataLoader(TensorDataset(X_eval, y_eval_tensor), batch_size=batch_size, shuffle=False)

        print(f"   → Optuna search (n_trials={n_trials_opt}) for {task_type} …")
        search_out = tune_hyperparameters(
            train_loader,
            val_loader,
            input_dim=input_dim,
            output_type=task_type,
            num_classes=num_classes,
            n_trials=n_trials_opt,
            device=device,
            val_split=val_split_opt,
            early_stopping_patience=patience_opt,
            results_dir=results_path,
            ordinal_sigma=ordinal_sigma,
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
            ordinal_sigma=ordinal_sigma,
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
        "latent":   latent_metrics if latent_metrics is not None else None,
    }

    print("Saving results …")
    eval.save_results(final_results, results_path)

    print(f"✅ Pipeline completed successfully to the path {results_path}!")
    

    
if __name__ == "__main__":
    main()
    


