import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances,
)
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns





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


def _to_tensor_2d(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() != 2:
        raise ValueError(f"Expected a 2-D tensor/array for latent features, got shape {tuple(t.shape)}")
    return t


def _cluster_metrics(z_np: np.ndarray, n_clusters: int = 5) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    n_samples = z_np.shape[0]
    if n_samples <= n_clusters:
        return metrics
    try:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(z_np)
        if len(np.unique(labels)) < 2:
            return metrics
        metrics["silhouette"] = float(silhouette_score(z_np, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(z_np, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z_np, labels))
    except Exception:
        # Be robust in case of degenerate inputs
        pass
    return metrics


def _trustworthiness(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Sklearn-like trustworthiness without dependency. Returns [0,1]."""
    # Rank distances in original space
    D_X = pairwise_distances(X_high)
    np.fill_diagonal(D_X, np.inf)
    ranks = D_X.argsort(axis=1)
    # Neighbours in low-d space
    D_Y = pairwise_distances(X_low)
    np.fill_diagonal(D_Y, np.inf)
    neigh_low = D_Y.argsort(axis=1)[:, :n_neighbors]
    n, k = X_high.shape[0], n_neighbors
    t_sum = 0.0
    for i in range(n):
        Ni_low = set(neigh_low[i])
        # ranks[i] lists neighbours from closest to farthest in high-d
        for j in Ni_low:
            # position (rank) of j in high-d list
            r_ij = int(np.where(ranks[i] == j)[0][0]) + 1  # 1-based
            if r_ij > k:
                t_sum += r_ij - k
    denom = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / denom) * t_sum if denom > 0 else 0.0


def _continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10) -> float:
    """Continuity measure [0,1]."""
    D_X = pairwise_distances(X_high)
    np.fill_diagonal(D_X, np.inf)
    neigh_high = D_X.argsort(axis=1)[:, :n_neighbors]
    D_Y = pairwise_distances(X_low)
    np.fill_diagonal(D_Y, np.inf)
    ranks_low = D_Y.argsort(axis=1)
    n, k = X_high.shape[0], n_neighbors
    c_sum = 0.0
    for i in range(n):
        for j in neigh_high[i]:
            r_ij = int(np.where(ranks_low[i] == j)[0][0]) + 1
            if r_ij > k:
                c_sum += r_ij - k
    denom = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / denom) * c_sum if denom > 0 else 0.0


def _distance_correlation(X_high: np.ndarray, X_low: np.ndarray) -> float:
    """Pearson correlation of vectorised pairwise distances (global geometry)."""
    Dh = pairwise_distances(X_high)
    Dl = pairwise_distances(X_low)
    iu = np.triu_indices_from(Dh, k=1)
    a, b = Dh[iu], Dl[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_latent_features(t_latent_features, e_latent_features, results_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Compute a practical subset of latent-evaluation metrics that do not require labels or original inputs.

    Implemented metrics:
      - active_units: number of dimensions with variance above a small threshold
      - variance_per_dim: per-dimension variance summary
      - hsic_matrix/global_score: independence via HSIC (saved heatmaps per split)
      - cluster quality: silhouette, Davies–Bouldin, Calinski–Harabasz on KMeans(k=5)
      - pca_explained_variance: PCA explained variance ratio on concatenated latents
    """
    #convert to numpy arrays, but only the first element of each sample
    train_latent_features = np.array([sample[0] for sample in t_latent_features.dataset])
    eval_latent_features = np.array([sample[0] for sample in e_latent_features.dataset])
    
    z_tr = _to_tensor_2d(train_latent_features)
    z_ev = _to_tensor_2d(eval_latent_features)

    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    train_dir = results_dir / "train"
    eval_dir = results_dir / "eval"
    train_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)
    # Try to extract labels from the datasets if available: (z, gender, age, abnormal)
    def _extract_labels(ds):
        ys = {"gender": [], "age": [], "abnormal": []}
        try:
            for sample in ds:
                ys["gender"].append(float(sample[1]))
                ys["age"].append(float(sample[2]))
                ys["abnormal"].append(float(sample[3]))
            return {k: np.asarray(v, dtype=float) for k, v in ys.items()}
        except Exception:
            return {"gender": None, "age": None, "abnormal": None}

    y_tr = _extract_labels(t_latent_features.dataset)
    y_ev = _extract_labels(e_latent_features.dataset)



    eps = 1e-6
    var_tr = z_tr.var(dim=0, unbiased=False).cpu().numpy()
    var_ev = z_ev.var(dim=0, unbiased=False).cpu().numpy()
    active_tr = int((var_tr > 1e-3).sum())
    active_ev = int((var_ev > 1e-3).sum())

    hsic_tr = independence_of_features(z_tr, save_path=str(train_dir), device="cpu")
    hsic_ev = independence_of_features(z_ev, save_path=str(eval_dir), device="cpu")

    # Cluster metrics (unsupervised)
    clus_tr = _cluster_metrics(z_tr.cpu().numpy(), n_clusters=5)
    clus_ev = _cluster_metrics(z_ev.cpu().numpy(), n_clusters=5)

    # PCA explained variance on concatenated latents
    try:
        z_all = torch.cat([z_tr, z_ev], dim=0).cpu().numpy()
        pca = PCA(n_components=min(z_all.shape[0], z_all.shape[1]))
        pca.fit(z_all)
        evr = pca.explained_variance_ratio_.astype(np.float32)
        # Summaries
        pca_summary = {
            "top5_ratio_sum": float(evr[:5].sum()) if evr.size >= 5 else float(evr.sum()),
            "explained_variance_ratio": evr,  # full vector
        }

        # Plot explained variance curve
        try:
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), marker="o")
            plt.xlabel("Number of components")
            plt.ylabel("Cumulative explained variance")
            plt.title("PCA explained variance (cumulative)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "pca_explained_variance_curve.png"))
            plt.close()
        except Exception:
            pass
    except Exception:
        pca_summary = {}

    # Geometry preservation metrics with a 2D PCA projection (deterministic)
    try:
        pca2 = PCA(n_components=2)
        z_tr_2d = pca2.fit_transform(z_tr.cpu().numpy())
        z_ev_2d = pca2.fit_transform(z_ev.cpu().numpy())
        geom_tr = {
            "trustworthiness": _trustworthiness(z_tr.cpu().numpy(), z_tr_2d, n_neighbors=10),
            "continuity": _continuity(z_tr.cpu().numpy(), z_tr_2d, n_neighbors=10),
            "dist_corr": _distance_correlation(z_tr.cpu().numpy(), z_tr_2d),
        }
        geom_ev = {
            "trustworthiness": _trustworthiness(z_ev.cpu().numpy(), z_ev_2d, n_neighbors=10),
            "continuity": _continuity(z_ev.cpu().numpy(), z_ev_2d, n_neighbors=10),
            "dist_corr": _distance_correlation(z_ev.cpu().numpy(), z_ev_2d),
        }
        # Save simple scatter plots
        plt.figure(figsize=(5, 4))
        plt.scatter(z_tr_2d[:, 0], z_tr_2d[:, 1], s=6, alpha=0.6)
        plt.title("PCA(2) – train")
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, "pca2_scatter.png"))
        plt.close()
        plt.figure(figsize=(5, 4))
        plt.scatter(z_ev_2d[:, 0], z_ev_2d[:, 1], s=6, alpha=0.6)
        plt.title("PCA(2) – eval")
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "pca2_scatter.png"))
        plt.close()

        # Optional t-SNE (subsample if needed)
        def _tsne_scatter(Z: np.ndarray, out_path: str):
            try:
                n = Z.shape[0]
                if n > 2000:
                    idx = np.random.RandomState(42).choice(n, size=2000, replace=False)
                    Z = Z[idx]
                perplexity = max(5, min(30, (Z.shape[0] - 1) // 3))
                tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
                Z2 = tsne.fit_transform(Z)
                plt.figure(figsize=(5, 4))
                plt.scatter(Z2[:, 0], Z2[:, 1], s=6, alpha=0.6)
                plt.title("t-SNE(2)")
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
            except Exception:
                pass

        _tsne_scatter(z_tr.cpu().numpy(), os.path.join(train_dir, "tsne_scatter.png"))
        _tsne_scatter(z_ev.cpu().numpy(), os.path.join(eval_dir, "tsne_scatter.png"))

        # Shepard plot: pairwise distances high-d vs 2D
        def _shepard(Z_hd: np.ndarray, Z_2d: np.ndarray, out_path: str):
            try:
                Dh = pairwise_distances(Z_hd)
                Dl = pairwise_distances(Z_2d)
                iu = np.triu_indices_from(Dh, k=1)
                a, b = Dh[iu], Dl[iu]
                m = a.shape[0]
                if m > 20000:
                    idx = np.random.RandomState(42).choice(m, size=20000, replace=False)
                    a = a[idx]
                    b = b[idx]
                plt.figure(figsize=(5, 4))
                plt.scatter(a, b, s=3, alpha=0.3)
                plt.xlabel("Distances in latent space")
                plt.ylabel("Distances in 2D embedding")
                plt.title("Shepard plot")
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
            except Exception:
                pass

        _shepard(z_tr.cpu().numpy(), z_tr_2d, os.path.join(train_dir, "shepard_plot.png"))
        _shepard(z_ev.cpu().numpy(), z_ev_2d, os.path.join(eval_dir, "shepard_plot.png"))
    except Exception:
        geom_tr, geom_ev = {}, {}

    metrics: Dict[str, Any] = {
        "train": {
            "n_samples": int(z_tr.shape[0]),
            "dim": int(z_tr.shape[1]),
            "variance_per_dim": var_tr,
            "active_units": active_tr,
            "hsic_global_score": float(hsic_tr.get("global_score", 0.0)),
            "cluster": clus_tr,
            "geometry": geom_tr,
        },
        "eval": {
            "n_samples": int(z_ev.shape[0]),
            "dim": int(z_ev.shape[1]),
            "variance_per_dim": var_ev,
            "active_units": active_ev,
            "hsic_global_score": float(hsic_ev.get("global_score", 0.0)),
            "cluster": clus_ev,
            "geometry": geom_ev,
        },
        "pca": pca_summary,
    }

    # Additional distribution plots: variance per dim
    try:
        plt.figure(figsize=(6, 3))
        plt.hist(var_tr, bins=30, alpha=0.7)
        plt.xlabel("Variance per latent dimension (train)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(train_dir, "variance_hist.png"))
        plt.close()
        plt.figure(figsize=(6, 3))
        plt.hist(var_ev, bins=30, alpha=0.7)
        plt.xlabel("Variance per latent dimension (eval)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, "variance_hist.png"))
        plt.close()
    except Exception:
        pass


    # Supervised mutual information I(Z;Y) per available target
    def _compute_mi(Z: np.ndarray, y: np.ndarray, task: str):
        if task in {"gender", "abnormal"}:
            yy = y.copy()
            uniq = np.unique(yy)
            if set(uniq).issubset({1.0, 2.0}):
                yy = (yy == 2.0).astype(float)
            if len(np.unique(yy)) < 2:
                return None
            return mutual_info_classif(Z, yy, random_state=42)
        else:  # age
            if np.std(y) < 1e-8:
                return None
            return mutual_info_regression(Z, y, random_state=42)

    try:
        for split_name, Z, Y in (
            ("train", z_tr.cpu().numpy(), y_tr),
            ("eval", z_ev.cpu().numpy(), y_ev),
        ):
            mi_out = {}
            for task in ("gender", "age", "abnormal"):
                yvec = Y.get(task)
                if yvec is None or len(yvec) != Z.shape[0]:
                    continue
                mi = _compute_mi(Z, yvec, task)
                if mi is None:
                    continue
                mi_out[task] = {
                    "mean": float(np.mean(mi)),
                    "per_dim": mi.astype(np.float32),
                }
            if mi_out:
                metrics[split_name]["mi_zy"] = mi_out
    except Exception:
        pass

    return metrics
