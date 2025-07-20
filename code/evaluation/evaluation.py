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

def evaluate_model(model: ClassificationModel, data, device: str = 'cpu'):
    """
    Evaluate the model on the given data.

    Parameters
    ----------
    model : ClassificationModel
        The trained classification model.
    data : DataLoader
        DataLoader with evaluation data.

    Returns
    -------
    dict
        Dictionary with mean loss metrics.
    """
    assert hasattr(model, "age_mean") and hasattr(model, "age_std"), "Model must have age_mean and age_std set for denormalization."
    model.eval()
    total_loss_g = 0.0
    total_loss_a = 0.0
    total_loss_abn = 0.0
    total_loss = 0.0
    total_samples = 0
    total_mae_a = 0.0
    age_true = []
    age_pred = []
    
    total_correct_g = 0
    total_correct_abn = 0

    g_true_all = []
    g_pred_all = []
    abn_true_all = []
    abn_pred_all = []

    with torch.no_grad():
        for x, g, a, ab in data:
            batch_size = x.size(0)
            total_samples += batch_size
            

            x, g, a, ab = x.to(device), g.to(device), a.to(device), ab.to(device)
            
            x = x.float()
            g = (g == 2).float()
            ab = ab.float()
            a = a.float()
            age_true.append(a.cpu())
            # print(f"Evaluating batch size: {batch_size}, x shape: {x}, g shape: {g}, a shape: {a}, ab shape: {ab}")
            
            ĝ, â, abn = model(x)
            â_denorm = model.denormalize_age(â)
            age_pred.append(â_denorm.cpu())
            
            # Compute accuracy for gender and abnormality
            g_pred = (ĝ > 0.5).float()
            abn_pred = (abn > 0.5).float()

            g_true_all.append(g.cpu())
            g_pred_all.append(g_pred.cpu())
            abn_true_all.append(ab.cpu())
            abn_pred_all.append(abn_pred.cpu())

            correct_g = (g_pred == g).sum().item()
            correct_abn = (abn_pred == ab).sum().item()

            total_correct_g += correct_g
            total_correct_abn += correct_abn

            loss_g = model.g_loss(ĝ, g)
            loss_a = model.a_loss(â, a)
            loss_abn = model.abn_loss(abn, ab)

            a_denorm = model.denormalize_age(a)
            mae_a = torch.abs(â_denorm - a_denorm).sum().item()
            total_mae_a += mae_a

            total_loss_g += loss_g.item() * batch_size
            total_loss_a += loss_a.item() * batch_size
            total_loss_abn += loss_abn.item() * batch_size
            total_loss += (loss_g + loss_a + loss_abn).item() * batch_size
            
    # Age bin performance
    age_true_all = torch.cat(age_true).numpy()
    age_pred_all = torch.cat(age_pred).numpy()

    age_bins = [0, 20, 40, 60, 80, 100]
    bin_mae = {}
    for i in range(len(age_bins) - 1):
        mask = (age_true_all >= age_bins[i]) & (age_true_all < age_bins[i + 1])
        if np.any(mask):
            bin_mae[f"{age_bins[i]}–{age_bins[i+1]}"] = np.mean(np.abs(age_true_all[mask] - age_pred_all[mask]))
        else:
            bin_mae[f"{age_bins[i]}–{age_bins[i+1]}"] = None

    # -------- Additional aggregated metrics --------------------------
    g_true_concat   = torch.cat(g_true_all).numpy()
    g_pred_concat   = torch.cat(g_pred_all).numpy()
    abn_true_concat = torch.cat(abn_true_all).numpy()
    abn_pred_concat = torch.cat(abn_pred_all).numpy()

    # confusion matrices (2×2) for gender / abnormality
    cm_gender = confusion_matrix(g_true_concat, g_pred_concat, labels=[0.,1.])
    cm_abn    = confusion_matrix(abn_true_concat, abn_pred_concat, labels=[0.,1.])

    # precision / recall / f1 (macro) for binary tasks
    prf_g = precision_recall_fscore_support(g_true_concat, g_pred_concat, average='binary', zero_division=0)
    prf_a = precision_recall_fscore_support(abn_true_concat, abn_pred_concat, average='binary', zero_division=0)

    # RMSE computed manually to support older scikit-learn versions
    rmse_a = float(np.sqrt(((age_true_all - age_pred_all) ** 2).mean()))

    return {
        'loss_g': total_loss_g / total_samples,
        'loss_a': total_loss_a / total_samples,
        'loss_abn': total_loss_abn / total_samples,
        'total_loss': total_loss / total_samples,
        'accuracy_g': total_correct_g / total_samples,
        'accuracy_abn': total_correct_abn / total_samples,
        'mae_a': total_mae_a / total_samples,
        'rmse_a': rmse_a,
        'gender_confusion': cm_gender.tolist(),
        'abn_confusion': cm_abn.tolist(),
        'gender_precision_recall_f1': {
            'precision': prf_g[0],
            'recall': prf_g[1],
            'f1': prf_g[2],
        },
        'abn_precision_recall_f1': {
            'precision': prf_a[0],
            'recall': prf_a[1],
            'f1': prf_a[2],
        },
        'age_bin_mae': bin_mae,
    }
    
    
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


def run_evaluation(model: ClassificationModel, data, save_path):
    """
    Run comprehensive evaluation, including independence testing.

    Parameters
    ----------
    model : ClassificationModel
        The trained classification model.
    data : DataLoader
        DataLoader with evaluation data.

    Returns
    -------
    dict
        Dictionary with evaluation metrics and independence score.
    """
    metrics = evaluate_model(model, data)

    # Concatenate x data across batches
    xs = torch.cat([batch[0] for batch in data], dim=0)

    independence_scores = independence_of_features(xs, save_path=save_path)
    metrics['global_independence_score'] = independence_scores['global_score']  # Keep only summary
    
    return metrics
