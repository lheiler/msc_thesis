import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from model_training.classification_model import ClassificationModel
import os
from typing import Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt



def independence_of_features(xs: torch.Tensor, save_path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Compute pair-wise dependence between latent dimensions using a
    biased Hilbert-Schmidt Independence Criterion (HSIC).

    * Each coordinate is z-scored.
    * Bandwidth per coordinate = median non-zero pairwise distance
      (“median heuristic”). Constant coordinates get HSIC = 0.
    * Returns:
        - 'hsic'  : (d,d) symmetric matrix (zeros on diagonal)
        - 'global_score' : float, mean off-diagonal HSIC
    """
    xs = xs.float().to(device)
    n, d = xs.shape

    # 1. z-score each coordinate
    xs_std = (xs - xs.mean(0, keepdim=True)) / xs.std(0, keepdim=True).clamp_min(1e-8)

    H = torch.eye(n, device=device) - 1.0 / n          # centring matrix
    Ks = []                                            # list of centred kernels

    for j in range(d):
        col = xs_std[:, j:j + 1]              # (n, 1)

        if col.std() < 1e-6:                  # constant feature
            Ks.append(torch.zeros(n, n, device=device))
            continue

        # --- fix: pair-wise squared distances (n × n) ---------------
        d2 = (col - col.T).pow(2)             # ← broadcasting, shape (n,n)

        # median of non-zero distances
        nz = d2[d2 > 0]
        sigma = torch.sqrt(0.5 * nz.median() + 1e-7) if nz.numel() else torch.tensor(1.0, device=device)

        K = torch.exp(-d2 / (2 * sigma ** 2)) # (n,n)
        Ks.append(H @ K @ H)
    Ks = torch.stack(Ks)                               # (d,n,n)

    # 2. biased HSIC
    hsic = torch.zeros(d, d, device=device)
    norm = (n - 1) ** 2
    for i in range(d):
        for j in range(i + 1, d):
            val = (Ks[i] * Ks[j]).sum() / norm
            hsic[i, j] = hsic[j, i] = val

    # 3. global score (mean off-diagonal, ignore zeros from constant dims)
    mask = hsic != 0
    global_score = hsic[mask].mean().item() if mask.any() else 0.0
    
    
    sns.heatmap(hsic.cpu(), vmin=0, vmax=0.05, square=True, cmap="mako")
    plt.savefig(os.path.join(save_path, "hsic_matrix.png"))
    plt.close()

    return {"hsic": hsic.cpu(), "global_score": global_score}

    
def _flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dicts for easier writing to text file."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def save_results(metrics, file_path: str):
    """Save evaluation metrics with inline explanations.

    Large arrays are summarised; each key is followed by a short description
    so readers know what the number means.
    """
    import os, numpy as _np

    _DESCR = {
        "loss_g": "Binary-cross-entropy loss for gender head (lower is better)",
        "loss_a": "MSE loss for age head after z-scaling", 
        "loss_abn": "Binary-cross-entropy loss for abnormality head",
        "total_loss": "Sum of all three task losses", 
        "accuracy_g": "Classification accuracy for gender (proportion correct)",
        "accuracy_abn": "Classification accuracy for abnormality",
        "mae_a": "Mean absolute error for age prediction (years)",
        "rmse_a": "Root-mean-square error for age prediction (years)",
        "gender_confusion": "2×2 confusion matrix: rows=true, cols=predicted (gender)",
        "abn_confusion": "2×2 confusion matrix: rows=true, cols=predicted (abnormality)",
        "gender_precision_recall_f1.precision": "Gender-task precision (positive=males)",
        "gender_precision_recall_f1.recall": "Gender-task recall",
        "gender_precision_recall_f1.f1": "Gender-task F1-score",
        "abn_precision_recall_f1.precision": "Abnormality precision (positive=abnormal)",
        "abn_precision_recall_f1.recall": "Abnormality recall",
        "abn_precision_recall_f1.f1": "Abnormality F1-score",
        "global_independence_score": "Mean off-diag HSIC of latent features (lower = more independent)",
        "age_bin_mae": "Dict: MAE per age bin (years)",
        "train_dataset_stats.n_samples": "Number of training samples",
        "eval_dataset_stats.n_samples": "Number of evaluation samples",
    }

    flat = _flatten_dict(metrics)

    with open(os.path.join(file_path, "final_metrics.txt"), "w") as f:
        for key, value in flat.items():
            if isinstance(value, _np.ndarray):
                val_repr = f"<array shape {value.shape}>"
            else:
                val_repr = value
            descr = _DESCR.get(key, "")
            if descr:
                f.write(f"{key}: {val_repr}    # {descr}\n")
            else:
                f.write(f"{key}: {val_repr}\n")

    # Also generate Markdown & JSON via utils.reporting
    try:
        from utils.reporting import write_markdown_report
        write_markdown_report(metrics, file_path)
    except Exception as e:
        print(f"⚠️ Could not write Markdown report: {e}")


