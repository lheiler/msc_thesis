from data_preprocessing import data_loading as dl
from evaluation.model_training.optuna_search import tune_hyperparameters
from evaluation.model_training.single_task_model import SingleTaskModel, train as train_single_task
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
from evaluation.cross_validation import CrossValidator, cv_results_to_dict, LogisticProbe
from pathlib import Path
import pickle


#todo: add timer for how long feature extraction takes


def main():
    """
    Main function to run the entire pipeline across multiple datasets.
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
    # 2) Core config and paths setup
    # ------------------------------------------------------------------
    method = cfg.get("method")
    
    # If --method is supplied, it overrides config
    method = args.method if args.method is not None else method

    paths_cfg = cfg.get("paths", {})
    results_root = paths_cfg.get("results_root", "Results")
    
    datasets = cfg.get("datasets", {})
    if not datasets:
        print("❌ No datasets configured in YAML. Expecting 'datasets' dictionary.")
        return

    for data_corp, dataset_path in datasets.items():
        data_path = os.path.expanduser(dataset_path)
        print(f"\n{'='*60}")
        print(f"🚀  Starting pipeline for dataset: {data_corp}")
        print(f"📂  Data path: {data_path}")
        print(f"{'='*60}\n")
        
        results_path = os.path.join(results_root, f"{data_corp}-{method}")
        os.makedirs(results_path, exist_ok=True)
        
        print(f"Results will be saved to: {results_path}")

        # ------------------------------------------------------------------
        # 3) Hyperparameters
        # ------------------------------------------------------------------
        optuna_cfg  = cfg.get("optuna", {})
        n_trials_opt   = optuna_cfg.get("n_trials", 30)
        val_split_opt  = optuna_cfg.get("val_split", 0.2)
        patience_opt   = optuna_cfg.get("patience", 10)
        batch_size   = optuna_cfg.get("batch_size", 64)
        
        # Load counts from pickle files
        train_pickle = os.path.join(data_path, "train_epochs.pkl")
        eval_pickle = os.path.join(data_path, "eval_epochs.pkl")
        
        if os.path.exists(train_pickle) and os.path.exists(eval_pickle):
            with open(train_pickle, 'rb') as f:
                n_train = len(pickle.load(f))
            with open(eval_pickle, 'rb') as f:
                n_eval = len(pickle.load(f))
        else:
            print(f"⚠️  Pickle files not found for {data_corp}. Skipping.")
            continue
        
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
                print("⚠️  Cache size mismatch – regenerating …")
                print(f"Expected {n_train} train samples, got {len(t_latent_features.dataset)}")
                print(f"Expected {n_eval} eval samples, got {len(e_latent_features.dataset)}")
                #use_cache = False ### CAREFULL THIS SHOULD BE REENABLED
            else: 
                print("Cached latent features loaded successfully.")

        if not use_cache:
            print("Loading and extracting latent features …")
            try:
                t_data = dl.load_data(data_path, "train")
                e_data = dl.load_data(data_path, "eval")
            except Exception as e:
                print(f"❌ Failed to load data: {e}")
                continue
                
            t_latent_features = extractor.extract_latent_features(
                t_data, batch_size=batch_size, method=method,
                save_path=os.path.join(results_path, "temp_latent_features_train.json"),
                dataset_name=data_corp
            )
            e_latent_features = extractor.extract_latent_features(
                e_data, batch_size=batch_size, method=method,
                save_path=os.path.join(results_path, "temp_latent_features_eval.json"),
                dataset_name=data_corp
            )
            
        # ------------------------------------------------------------------
        # 5) Latent evaluation
        # ------------------------------------------------------------------
        print(f"\n{'─'*60}")
        print(f"📊  PHASE 1: Latent Feature Evaluation")
        print(f"{'─'*60}")
        latent_metrics_file = os.path.join(results_path, "latent_metrics.json")
        if not reset and os.path.exists(latent_metrics_file):
            with open(latent_metrics_file, "r") as f:
                latent_metrics = json.load(f)
        else:
            try:
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray): return obj.tolist()
                        if isinstance(obj, np.generic): return obj.item()
                        return super().default(obj)
                        
                latent_metrics = metrics.evaluate_latent_features(t_latent_features, e_latent_features, results_path)
                with open(latent_metrics_file, "w") as f:
                    json.dump(latent_metrics, f, indent=4, cls=NumpyEncoder)
            except Exception as e:
                print(f"⚠️ Latent evaluation failed: {e}")
                latent_metrics = None

        # ------------------------------------------------------------------
        # 6) Training setup
        # ------------------------------------------------------------------
        print(f"\n{'─'*60}")
        print(f"⚙️  PHASE 2: Training Setup")
        print(f"{'─'*60}")
        input_dim = t_latent_features.dataset[0][0].numel()
        metrics_all = {}
        hyperparams_all = {}
        cv_results_all = {}
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        # Subject-wise split helper
        def _extract_subject_id(sample_id: str) -> str:
            m_tuh = re.match(r"^([A-Za-z0-9]+)_s\d+", sample_id)
            if m_tuh: return m_tuh.group(1)
            m_bids = re.match(r"^(sub-[A-Za-z0-9]+)", sample_id)
            if m_bids: return m_bids.group(1)
            for mkr in ["_s", "_t", "_epoch"]:
                if mkr in sample_id: return sample_id.split(mkr, 1)[0]
            if "_" in sample_id: return sample_id.split("_", 1)[0]
            return sample_id

        sample_ids_train = getattr(t_latent_features, "sample_ids", None)
        if sample_ids_train and len(sample_ids_train) == len(t_latent_features.dataset):
            subject_groups = [_extract_subject_id(sid) for sid in sample_ids_train]
            gss = GroupShuffleSplit(n_splits=1, test_size=val_split_opt, random_state=42)
            train_indices_global, val_indices_global = next(gss.split(list(range(len(subject_groups))), groups=subject_groups))
            print(f"Subject-wise split: {len(set([subject_groups[i] for i in train_indices_global]))} train | "
                  f"{len(set([subject_groups[i] for i in val_indices_global]))} val subjects")
        else:
            train_indices_global, val_indices_global = train_test_split(
                list(range(len(t_latent_features.dataset))), test_size=val_split_opt, random_state=42
            )
            print("WARNING: sample_ids missing; using random epoch split.")

        # Define tasks
        task_map = {}
        if data_corp == "lemon":
            task_map[0] = ("regression", "age", 2)
        elif data_corp in ("tuh", "harvard"):
            task_map[0] = ("classification", "abnormal", 3)
        else:
            task_map[0] = ("regression", "age", 2)
            task_map[1] = ("classification", "abnormal", 3)

        def build_xy(dataset, target_idx):
            X = torch.stack([s[0].detach().clone().float() for s in dataset])
            y = torch.tensor([float(s[target_idx]) for s in dataset], dtype=torch.float32)
            return X, y

        def map_class_labels(y_tensor):
            if torch.all((y_tensor == 1) | (y_tensor == 2)): return (y_tensor == 1).float()
            if torch.all((y_tensor == 0) | (y_tensor == 1)): return y_tensor.float()
            return y_tensor

        def discretize_age(y_tensor):
            """Binary: 0 = Young (<45), 1 = Old (>=45).
            The LEMON dataset has a bimodal distribution around these two cohorts,
            with the gap naturally falling around 45 years."""
            return (y_tensor >= 45).float()

        for task_idx in range(len(task_map)):
            task_type, task_name, tuple_idx = task_map[task_idx]
            num_classes, ordinal_sigma = 1, None
            
            if data_corp == "lemon" and task_name == "age":
                task_type, num_classes, ordinal_sigma = "classification", 1, None
                print(f"🔹 Task {task_idx+1}: [LEMON] Age Binary Classification (Young <45 vs Old ≥45)")
            else:
                print(f"🔹 Task {task_idx+1}: {task_name} ({task_type})")

            # Build train
            X_train, y_train = build_xy(t_latent_features.dataset, tuple_idx)
            if task_type == "classification":
                y_train = discretize_age(y_train) if (data_corp == "lemon" and task_name == "age") else map_class_labels(y_train)

            # CV: 5-fold subject-wise cross-validation on TRAINING data
            print(f"\n{'─'*60}")
            print(f"🔄  PHASE 3: Cross-Validation ({task_name}) — 5-fold on TRAINING data")
            print(f"{'─'*60}")
            cv = CrossValidator(n_splits=5, n_trials=n_trials_opt, batch_size=batch_size, device=device)
            cv_result = cv.run(
                X=X_train.numpy(), y=y_train.numpy(), 
                sample_ids=sample_ids_train or [str(i) for i in range(len(X_train))],
                task_type=task_type, num_classes=num_classes, task_name=task_name,
                ordinal_sigma=ordinal_sigma, results_dir=results_path
            )
            cv_results_all[task_name] = cv_result

            # Normalization (Train split)
            train_tensor_idx = torch.tensor(train_indices_global)
            X_mean = X_train[train_tensor_idx].mean(0, keepdim=True)
            X_std = X_train[train_tensor_idx].std(0, keepdim=True) + 1e-8
            X_train_norm = (X_train - X_mean) / X_std

            train_loader = DataLoader(Subset(TensorDataset(X_train_norm, y_train), train_indices_global), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(Subset(TensorDataset(X_train_norm, y_train), val_indices_global), batch_size=batch_size, shuffle=False)

            # Build eval
            X_eval, y_eval = build_xy(e_latent_features.dataset, tuple_idx)
            if task_type == "classification":
                y_eval = discretize_age(y_eval) if (data_corp == "lemon" and task_name == "age") else map_class_labels(y_eval)
            X_eval_norm = (X_eval - X_mean) / X_std
            eval_loader = DataLoader(TensorDataset(X_eval_norm, y_eval), batch_size=batch_size, shuffle=False)

            # Reuse architecture discovered during CV (fold 0 Optuna)
            best_arch = cv_results_all[task_name]["best_architecture"]
            best_params = cv_results_all[task_name]["best_optuna_params"]

            print(f"\n{'─'*60}")
            print(f"🏗️  PHASE 4: Retrain CV architecture ({task_name}) — on full TRAINING set")
            print(f"    Architecture: {best_arch['hidden_dims']}, dropout={best_arch['dropout']:.2f}")
            print(f"    lr={best_params.get('lr', 1e-3):.5f}, wd={best_params.get('weight_decay', 1e-4):.6f}, sched={best_params.get('scheduler', 'plateau')}")
            print(f"{'─'*60}")

            model = SingleTaskModel(
                input_dim=input_dim,
                output_type=task_type,
                hidden_dims=tuple(best_arch["hidden_dims"]),
                dropout=best_arch["dropout"],
                num_classes=best_arch.get("num_classes", num_classes),
            )
            train_single_task(
                model,
                train_loader,
                val_loader=val_loader,
                n_epochs=300,
                lr=best_params.get("lr", 1e-3),
                weight_decay=best_params.get("weight_decay", 1e-4),
                device=device,
                scheduler=best_params.get("scheduler", "plateau"),
                early_stopping_patience=patience_opt,
                ordinal_sigma=ordinal_sigma,
            )
            hyperparams_all[task_name] = best_params

            # Final Eval
            print(f"\n{'─'*60}")
            print(f"🎯  PHASE 5: Final Evaluation ({task_name}) — on held-out EVAL set")
            print(f"{'─'*60}")
            metrics_all[task_name] = model.evaluate(
                eval_loader, output_type=task_type, device=device, 
                plot_dir=os.path.join(results_path, f"plots_{task_name}"),
                ordinal_sigma=ordinal_sigma
            )

            # Linear probe baseline on eval set
            print(f"\n{'─'*60}")
            print(f"📏  PHASE 5b: Linear Probe Baseline ({task_name}) — on held-out EVAL set")
            print(f"{'─'*60}")
            probe = LogisticProbe(task_type=task_type, num_classes=num_classes)
            X_train_np = X_train_norm[train_tensor_idx].numpy()
            y_train_np = y_train[train_tensor_idx].numpy()
            probe.fit(X_train_np, y_train_np)
            probe_eval_metrics = probe.evaluate(X_eval_norm.numpy(), y_eval.numpy())
            metrics_all[f"{task_name}_linear_probe"] = probe_eval_metrics
            print(f"  Linear probe eval: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in probe_eval_metrics.items()
                              if isinstance(v, (int, float))))

        # Persist
        print(f"\n{'─'*60}")
        print(f"💾  PHASE 6: Saving Results")
        print(f"{'─'*60}")
        final_results = {
            "metrics_per_task": metrics_all,
            "hyperparams_per_task": hyperparams_all,
            "train_dataset_stats": compute_dataset_stats(t_latent_features),
            "eval_dataset_stats": compute_dataset_stats(e_latent_features),
            "latent": latent_metrics
        }
        final_results["cross_validation"] = {k: cv_results_to_dict(v, k) for k, v in cv_results_all.items()}
            
        eval.save_results(final_results, results_path)
        print(f"✅ Dataset {data_corp} done!")

    print("\n✅ All datasets processed successfully!")

if __name__ == "__main__":
    main()
