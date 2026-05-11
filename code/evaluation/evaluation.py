"""Results serialisation for the evaluation pipeline.

Writes a human-readable ``final_metrics.txt`` with inline descriptions
of each metric key.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

_DESCR: dict[str, str] = {
    "train.active_units": "Number of latent dimensions with variance > 1e-3 (train)",
    "eval.active_units": "Number of latent dimensions with variance > 1e-3 (eval)",
    "train.hsic_global_score": "Mean off-diagonal HSIC (lower = more independent) on train",
    "eval.hsic_global_score": "Mean off-diagonal HSIC (lower = more independent) on eval",
    "train.cluster.silhouette": "Silhouette score on KMeans(k=5) clusters (higher better)",
    "train.cluster.davies_bouldin": "Davies-Bouldin index (lower better)",
    "train.cluster.calinski_harabasz": "Calinski-Harabasz score (higher better)",
    "eval.cluster.silhouette": "Silhouette score on KMeans(k=5) clusters (higher better)",
    "eval.cluster.davies_bouldin": "Davies-Bouldin index (lower better)",
    "eval.cluster.calinski_harabasz": "Calinski-Harabasz score (higher better)",
    "train.geometry.trustworthiness": "Neighbourhood preservation in 2D PCA (train)",
    "train.geometry.continuity": "Lost-neighbour score complement in 2D PCA (train)",
    "train.geometry.dist_corr": "Correlation of pairwise distances in 2D PCA (train)",
    "eval.geometry.trustworthiness": "Neighbourhood preservation in 2D PCA (eval)",
    "eval.geometry.continuity": "Lost-neighbour score complement in 2D PCA (eval)",
    "eval.geometry.dist_corr": "Correlation of pairwise distances in 2D PCA (eval)",
    "pca.top5_ratio_sum": "Sum of first 5 explained variance ratios of PCA(Z)",
    "metrics_per_task.abnormal.accuracy": "Classification accuracy (abnormal)",
    "metrics_per_task.abnormal.f1": "Binary F1-score (abnormal)",
    "metrics_per_task.abnormal.f1_macro": "Macro-averaged F1 (abnormal)",
    "metrics_per_task.abnormal.roc_auc": "ROC-AUC (abnormal)",
    "metrics_per_task.abnormal.pr_auc": "Precision-Recall AUC (abnormal)",
    "metrics_per_task.gender.accuracy": "Classification accuracy (gender)",
    "metrics_per_task.gender.f1": "Binary F1-score (gender)",
    "metrics_per_task.gender.f1_macro": "Macro-averaged F1 (gender)",
    "metrics_per_task.gender.roc_auc": "ROC-AUC (gender)",
    "metrics_per_task.gender.pr_auc": "Precision-Recall AUC (gender)",
    "metrics_per_task.age.mae": "Mean absolute error (age)",
    "metrics_per_task.age.rmse": "Root-mean-square error (age)",
    "metrics_per_task.age.r2": "Coefficient of determination R2 (age)",
    "train_dataset_stats.n_samples": "Number of training samples",
    "eval_dataset_stats.n_samples": "Number of evaluation samples",
    "cross_validation": "5-fold subject-wise cross-validation results (MLP + linear probe)",
}


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for all keys.
        sep: Separator between nested keys.

    Returns:
        Flat dictionary.
    """
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def save_results(metrics: dict[str, Any], file_path: str) -> None:
    """Write evaluation metrics to ``final_metrics.txt``.

    Large arrays are summarised; each key is annotated with a short
    description.

    Args:
        metrics: Nested metrics dictionary.
        file_path: Directory path where the file will be saved.
    """
    flat = _flatten_dict(metrics)

    with open(os.path.join(file_path, "final_metrics.txt"), "w") as f:
        for key, value in flat.items():
            if isinstance(value, np.ndarray):
                val_repr = f"<array shape {value.shape}>"
            else:
                val_repr = value
            descr = _DESCR.get(key, "")
            if descr:
                f.write(f"{key}: {val_repr}    # {descr}\n")
            else:
                f.write(f"{key}: {val_repr}\n")
