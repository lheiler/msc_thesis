"""EEG latent-feature extraction and classification pipeline.

Orchestrates: config loading -> data loading -> latent extraction ->
unsupervised evaluation -> cross-validation -> final training -> results.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from typing import Any, Optional

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from data_preprocessing import data_loading as dl
from data_preprocessing.cache_loading import load_latent_parameters_array
from evaluation.cross_validation import CrossValidator, LogisticProbe, cv_results_to_dict
from evaluation.data_metrics import compute_dataset_stats
import evaluation.evaluation as eval_module
import evaluation.metrix as metrics
from evaluation.model_training.single_task_model import (
    SingleTaskModel,
    train as train_single_task,
)
import latent_extraction.extractor as extractor

logger = logging.getLogger(__name__)

TUH_BATCH_SIZE_MULTIPLIER = 2
AGE_BINARIZATION_THRESHOLD = 45.0
N_CV_FOLDS = 5
MAX_TRAIN_EPOCHS = 300


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and scalars."""

    def default(self, obj: object) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _extract_subject_id(sample_id: str) -> str:
    """Extract the subject identifier from a sample ID string.

    Args:
        sample_id: Raw sample identifier (e.g. ``"sub-001_s01_t01"``).

    Returns:
        Subject-level identifier stripped of session/epoch suffixes.
    """
    m_tuh = re.match(r"^([A-Za-z0-9]+)_s\d+", sample_id)
    if m_tuh:
        return m_tuh.group(1)
    m_bids = re.match(r"^(sub-[A-Za-z0-9]+)", sample_id)
    if m_bids:
        return m_bids.group(1)
    for marker in ["_s", "_t", "_epoch"]:
        if marker in sample_id:
            return sample_id.split(marker, 1)[0]
    if "_" in sample_id:
        return sample_id.split("_", 1)[0]
    return sample_id


def _build_xy(
    dataset: torch.utils.data.Dataset,
    target_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack feature vectors and target values from a dataset.

    Args:
        dataset: Dataset where each sample is a tuple of tensors.
        target_idx: Positional index of the target value in each tuple.

    Returns:
        Tuple of ``(X, y)`` tensors.
    """
    X = torch.stack([s[0].detach().clone().float() for s in dataset])
    y = torch.tensor(
        [float(s[target_idx]) for s in dataset], dtype=torch.float32
    )
    return X, y


def _map_class_labels(y_tensor: torch.Tensor) -> torch.Tensor:
    """Remap class labels to {0, 1} float tensor.

    Args:
        y_tensor: Raw label tensor (may use 1/2 encoding).

    Returns:
        Float tensor with binary labels.
    """
    if torch.all((y_tensor == 1) | (y_tensor == 2)):
        return (y_tensor == 1).float()
    if torch.all((y_tensor == 0) | (y_tensor == 1)):
        return y_tensor.float()
    return y_tensor


def _discretize_age(y_tensor: torch.Tensor) -> torch.Tensor:
    """Binarize age into Young (0) vs Old (1).

    Args:
        y_tensor: Continuous age values.

    Returns:
        Binary tensor (0 = young, 1 = old) split at
        ``AGE_BINARIZATION_THRESHOLD``.
    """
    return (y_tensor >= AGE_BINARIZATION_THRESHOLD).float()


def main() -> None:
    """Run the full EEG classification pipeline across configured datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # 1) Parse CLI arguments and load configuration
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="EEG classification pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the pipeline"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method to use for latent feature extraction",
    )
    args = parser.parse_args()
    reset: bool = args.reset

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # 2) Core config and paths setup
    # ------------------------------------------------------------------
    method: str = args.method if args.method is not None else cfg.get("method")

    paths_cfg: dict[str, str] = cfg.get("paths", {})
    results_root: str = paths_cfg.get("results_root", "Results")

    datasets: dict[str, str] = cfg.get("datasets", {})
    if not datasets:
        logger.error("No datasets configured in YAML. Expecting 'datasets' dictionary.")
        return

    for data_corp, dataset_path in datasets.items():
        data_path = os.path.expanduser(dataset_path)
        logger.info("=" * 60)
        logger.info("Starting pipeline for dataset: %s", data_corp)
        logger.info("Data path: %s", data_path)
        logger.info("=" * 60)

        results_path = os.path.join(results_root, f"{data_corp}-{method}")
        os.makedirs(results_path, exist_ok=True)

        logger.info("Results will be saved to: %s", results_path)

        # ------------------------------------------------------------------
        # 3) Hyperparameters
        # ------------------------------------------------------------------
        optuna_cfg: dict[str, Any] = cfg.get("optuna", {})
        n_trials_opt: int = optuna_cfg.get("n_trials", 30)
        val_split_opt: float = optuna_cfg.get("val_split", 0.2)
        patience_opt: int = optuna_cfg.get("patience", 10)
        batch_size: int = optuna_cfg.get("batch_size", 64)

        if data_corp == "tuh":
            batch_size *= TUH_BATCH_SIZE_MULTIPLIER
            logger.info("TUH batch size scaled to %d", batch_size)

        train_pickle = os.path.join(data_path, "train_epochs.pkl")
        eval_pickle = os.path.join(data_path, "eval_epochs.pkl")

        if os.path.exists(train_pickle) and os.path.exists(eval_pickle):
            with open(train_pickle, "rb") as f:
                n_train = len(pickle.load(f))
            with open(eval_pickle, "rb") as f:
                n_eval = len(pickle.load(f))
        else:
            logger.warning("Pickle files not found for %s. Skipping.", data_corp)
            continue

        # ------------------------------------------------------------------
        # 4) Latent feature loading: cache or compute
        # ------------------------------------------------------------------
        def _latent_loader(split: str) -> DataLoader:
            return load_latent_parameters_array(
                os.path.join(results_path, f"temp_latent_features_{split}"),
                batch_size=batch_size,
            )

        train_cache = os.path.join(results_path, "temp_latent_features_train.json")
        eval_cache = os.path.join(results_path, "temp_latent_features_eval.json")
        use_cache = not reset and os.path.exists(train_cache) and os.path.exists(eval_cache)

        if use_cache:
            t_latent_features = _latent_loader("train")
            e_latent_features = _latent_loader("eval")
            if (
                len(t_latent_features.dataset) != n_train
                or len(e_latent_features.dataset) != n_eval
            ):
                logger.warning(
                    "Cache size mismatch (expected %d/%d, got %d/%d) - regenerating",
                    n_train, n_eval,
                    len(t_latent_features.dataset),
                    len(e_latent_features.dataset),
                )
                use_cache = False
            else:
                logger.info("Cached latent features loaded successfully.")

        if not use_cache:
            logger.info("Loading and extracting latent features ...")
            try:
                t_data = dl.load_data(data_path, "train")
                e_data = dl.load_data(data_path, "eval")
            except (FileNotFoundError, IOError) as exc:
                logger.error("Failed to load data: %s", exc)
                continue

            t_latent_features = extractor.extract_latent_features(
                t_data,
                batch_size=batch_size,
                method=method,
                save_path=os.path.join(results_path, "temp_latent_features_train.json"),
                dataset_name=data_corp,
            )
            e_latent_features = extractor.extract_latent_features(
                e_data,
                batch_size=batch_size,
                method=method,
                save_path=os.path.join(results_path, "temp_latent_features_eval.json"),
                dataset_name=data_corp,
            )

        # ------------------------------------------------------------------
        # 5) Latent evaluation
        # ------------------------------------------------------------------
        logger.info("PHASE 1: Latent Feature Evaluation")
        latent_metrics_file = os.path.join(results_path, "latent_metrics.json")
        latent_metrics: Optional[dict[str, Any]] = None

        if not reset and os.path.exists(latent_metrics_file):
            with open(latent_metrics_file, "r") as f:
                latent_metrics = json.load(f)
        else:
            try:
                latent_metrics = metrics.evaluate_latent_features(
                    t_latent_features, e_latent_features, results_path
                )
                with open(latent_metrics_file, "w") as f:
                    json.dump(latent_metrics, f, indent=4, cls=_NumpyEncoder)
            except Exception as exc:
                logger.warning("Latent evaluation failed: %s", exc)

        # ------------------------------------------------------------------
        # 6) Training setup
        # ------------------------------------------------------------------
        logger.info("PHASE 2: Training Setup")
        input_dim: int = t_latent_features.dataset[0][0].numel()
        metrics_all: dict[str, Any] = {}
        hyperparams_all: dict[str, Any] = {}
        cv_results_all: dict[str, Any] = {}
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        sample_ids_train = getattr(t_latent_features, "sample_ids", None)
        if sample_ids_train and len(sample_ids_train) == len(t_latent_features.dataset):
            subject_groups = [_extract_subject_id(sid) for sid in sample_ids_train]
            gss = GroupShuffleSplit(n_splits=1, test_size=val_split_opt, random_state=42)
            train_indices_global, val_indices_global = next(
                gss.split(list(range(len(subject_groups))), groups=subject_groups)
            )
            n_train_subj = len({subject_groups[i] for i in train_indices_global})
            n_val_subj = len({subject_groups[i] for i in val_indices_global})
            logger.info(
                "Subject-wise split: %d train | %d val subjects",
                n_train_subj, n_val_subj,
            )
        else:
            train_indices_global, val_indices_global = train_test_split(
                list(range(len(t_latent_features.dataset))),
                test_size=val_split_opt,
                random_state=42,
            )
            logger.warning("sample_ids missing; using random epoch split.")

        # Define tasks per dataset
        task_map: dict[int, tuple[str, str, int]] = {}
        if data_corp == "lemon":
            task_map[0] = ("regression", "age", 2)
        elif data_corp in ("tuh", "harvard"):
            task_map[0] = ("classification", "abnormal", 3)
        else:
            task_map[0] = ("regression", "age", 2)
            task_map[1] = ("classification", "abnormal", 3)

        for task_idx in range(len(task_map)):
            task_type, task_name, tuple_idx = task_map[task_idx]
            num_classes: int = 1
            ordinal_sigma: Optional[float] = None

            if data_corp == "lemon" and task_name == "age":
                task_type = "classification"
                logger.info(
                    "Task %d: [LEMON] Age Binary Classification (Young <%d vs Old >=%d)",
                    task_idx + 1, int(AGE_BINARIZATION_THRESHOLD), int(AGE_BINARIZATION_THRESHOLD),
                )
            else:
                logger.info("Task %d: %s (%s)", task_idx + 1, task_name, task_type)

            # Build train
            X_train, y_train = _build_xy(t_latent_features.dataset, tuple_idx)
            if task_type == "classification":
                y_train = (
                    _discretize_age(y_train)
                    if (data_corp == "lemon" and task_name == "age")
                    else _map_class_labels(y_train)
                )

            # CV: 5-fold subject-wise cross-validation on TRAINING data
            logger.info(
                "PHASE 3: Cross-Validation (%s) - %d-fold on TRAINING data",
                task_name, N_CV_FOLDS,
            )
            cv = CrossValidator(
                n_splits=N_CV_FOLDS,
                n_trials=n_trials_opt,
                batch_size=batch_size,
                device=device,
            )
            cv_result = cv.run(
                X=X_train.numpy(),
                y=y_train.numpy(),
                sample_ids=sample_ids_train or [str(i) for i in range(len(X_train))],
                task_type=task_type,
                num_classes=num_classes,
                task_name=task_name,
                ordinal_sigma=ordinal_sigma,
                results_dir=results_path,
            )
            cv_results_all[task_name] = cv_result

            # Normalization (Train split)
            train_tensor_idx = torch.tensor(train_indices_global)
            X_mean = X_train[train_tensor_idx].mean(0, keepdim=True)
            X_std = X_train[train_tensor_idx].std(0, keepdim=True) + 1e-8
            X_train_norm = (X_train - X_mean) / X_std

            train_loader = DataLoader(
                Subset(TensorDataset(X_train_norm, y_train), train_indices_global),
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                Subset(TensorDataset(X_train_norm, y_train), val_indices_global),
                batch_size=batch_size,
                shuffle=False,
            )

            # Build eval
            X_eval, y_eval = _build_xy(e_latent_features.dataset, tuple_idx)
            if task_type == "classification":
                y_eval = (
                    _discretize_age(y_eval)
                    if (data_corp == "lemon" and task_name == "age")
                    else _map_class_labels(y_eval)
                )
            X_eval_norm = (X_eval - X_mean) / X_std
            eval_loader = DataLoader(
                TensorDataset(X_eval_norm, y_eval),
                batch_size=batch_size,
                shuffle=False,
            )

            # Reuse architecture discovered during CV (fold 0 Optuna)
            best_arch = cv_results_all[task_name]["best_architecture"]
            best_params = cv_results_all[task_name]["best_optuna_params"]

            logger.info(
                "PHASE 4: Retrain CV architecture (%s) on full TRAINING set | "
                "arch=%s dropout=%.2f lr=%.5f wd=%.6f sched=%s",
                task_name,
                best_arch["hidden_dims"],
                best_arch["dropout"],
                best_params.get("lr", 1e-3),
                best_params.get("weight_decay", 1e-4),
                best_params.get("scheduler", "plateau"),
            )

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
                n_epochs=MAX_TRAIN_EPOCHS,
                lr=best_params.get("lr", 1e-3),
                weight_decay=best_params.get("weight_decay", 1e-4),
                device=device,
                scheduler=best_params.get("scheduler", "plateau"),
                early_stopping_patience=patience_opt,
                ordinal_sigma=ordinal_sigma,
            )
            hyperparams_all[task_name] = best_params

            # Final Eval
            logger.info("PHASE 5: Final Evaluation (%s) on held-out EVAL set", task_name)
            metrics_all[task_name] = model.evaluate(
                eval_loader,
                output_type=task_type,
                device=device,
                plot_dir=os.path.join(results_path, f"plots_{task_name}"),
                ordinal_sigma=ordinal_sigma,
            )

            # Linear probe baseline on eval set
            logger.info("PHASE 5b: Linear Probe Baseline (%s) on held-out EVAL set", task_name)
            probe = LogisticProbe(task_type=task_type, num_classes=num_classes)
            X_train_np = X_train_norm[train_tensor_idx].numpy()
            y_train_np = y_train[train_tensor_idx].numpy()
            probe.fit(X_train_np, y_train_np)
            probe_eval_metrics = probe.evaluate(X_eval_norm.numpy(), y_eval.numpy())
            metrics_all[f"{task_name}_linear_probe"] = probe_eval_metrics
            logger.info(
                "Linear probe eval: %s",
                ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in probe_eval_metrics.items()
                    if isinstance(v, (int, float))
                ),
            )

        # Persist
        logger.info("PHASE 6: Saving Results")
        final_results: dict[str, Any] = {
            "metrics_per_task": metrics_all,
            "hyperparams_per_task": hyperparams_all,
            "train_dataset_stats": compute_dataset_stats(t_latent_features),
            "eval_dataset_stats": compute_dataset_stats(e_latent_features),
            "latent": latent_metrics,
        }
        final_results["cross_validation"] = {
            k: cv_results_to_dict(v, k) for k, v in cv_results_all.items()
        }

        eval_module.save_results(final_results, results_path)
        logger.info("Dataset %s done.", data_corp)

    logger.info("All datasets processed successfully.")


if __name__ == "__main__":
    main()
