import torch
from Model_Training.classification_model import ClassificationModel
from itertools import combinations
import os
from typing import Dict, Tuple
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
    
 
    sns.heatmap(hsic, vmin=0, vmax=0.05, square=True, cmap="mako")
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
    model.eval()
    total_loss_g = 0.0
    total_loss_a = 0.0
    total_loss_abn = 0.0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, g, a, ab in data:
            batch_size = x.size(0)
            total_samples += batch_size
            

            x, g, a, ab = x.to(device), g.to(device), a.to(device), ab.to(device)
            
            x = x.float()
            g = (g == 2).float()
            ab = ab.float()
            a = a.float()
            # print(f"Evaluating batch size: {batch_size}, x shape: {x}, g shape: {g}, a shape: {a}, ab shape: {ab}")
            
            ĝ, â, abn = model(x)

            loss_g = model.g_loss(ĝ, g)
            loss_a = model.a_loss(â, a)
            loss_abn = model.abn_loss(abn, ab)

            total_loss_g += loss_g.item() * batch_size
            total_loss_a += loss_a.item() * batch_size
            total_loss_abn += loss_abn.item() * batch_size
            total_loss += (loss_g + loss_a + loss_abn).item() * batch_size

    return {
        'loss_g': total_loss_g / total_samples,
        'loss_a': total_loss_a / total_samples,
        'loss_abn': total_loss_abn / total_samples,
        'total_loss': total_loss / total_samples
    }
    
    
def save_results(metrics, file_path: str):
    """
    Save evaluation metrics to a file.

    Parameters
    ----------
    metrics : dict
        Dictionary containing evaluation metrics.
    file_path : str
        Path to the file where metrics will be saved.
    """
    with open(os.path.join(file_path, "final_metrics.txt"), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    

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
    metrics['global_independence_score'] = independence_scores['global_score']
    metrics['hsic_matrix'] = independence_scores['hsic'].numpy()
    if 'pval' in independence_scores:
        metrics['pval_matrix'] = independence_scores['pval'].numpy()
    else:
        metrics['pval_matrix'] = None
    
    return metrics
