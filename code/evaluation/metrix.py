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
from scipy import stats
from scipy.stats import pearsonr

# Set up consistent plot styling
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
    'grid.linewidth': 0.5,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})





def independence_of_features(xs: torch.Tensor, save_path, device: str = "cpu", max_samples: int = 10000) -> Dict[str, torch.Tensor]:
    """
    Compute pair-wise dependence between latent dimensions using a
    biased Hilbert-Schmidt Independence Criterion (HSIC).

    * Each coordinate is z-scored.
    * Bandwidth per coordinate = median non-zero pairwise distance
      ("median heuristic"). Constant coordinates get HSIC = 0.
    * Returns:
        - 'hsic'  : (d,d) symmetric matrix (zeros on diagonal)
        - 'global_score' : float, mean off-diagonal HSIC
    
    Args:
        max_samples: Maximum number of samples to use (subsamples if needed for efficiency)
    """
    xs = xs.float().to(device)
    n, d = xs.shape
    
    # Subsample for computational efficiency if dataset is large
    original_n = n
    if n > max_samples:
        print(f"HSIC: Subsampling {max_samples:,} from {n:,} samples for efficiency")
        # Use seeded random sampling for reproducibility
        torch.manual_seed(42)
        idx = torch.randperm(n, device=device)[:max_samples]
        xs = xs[idx]
        n = max_samples

    # 1. z-score each coordinate
    xs_std = (xs - xs.mean(0, keepdim=True)) / xs.std(0, keepdim=True).clamp_min(1e-8)

    # Implicit centering helper: HKH = K - row_mean - col_mean + global_mean
    def _center_kernel_inplace(K: torch.Tensor) -> None:
        row_mean = K.mean(dim=1, keepdim=True)
        col_mean = K.mean(dim=0, keepdim=True)
        global_mean = K.mean()
        K -= row_mean
        K -= col_mean
        K += global_mean

    # Compute HSIC exactly using block-wise kernels to keep memory bounded.
    # No approximation: we compute the same centered Gaussian kernels and exact Frobenius inner products.
    hsic = torch.zeros(d, d, device=device)
    norm = (n - 1) ** 2

    # Fixed block size chosen to fit ~419 dims with n≈10,000 on 24GB VRAM
    block_dim = 16

    # Precompute standardized columns for stability
    cols = [xs_std[:, j:j + 1] for j in range(d)]
    const_dim = [bool(col.std() < 1e-6) for col in cols]

    # Process dimensions in blocks
    for i_start in range(0, d, block_dim):
        i_end = min(d, i_start + block_dim)
        # Build and center kernels for i-block
        Ki_list = []
        for i in range(i_start, i_end):
            if const_dim[i]:
                Ki = torch.zeros(n, n, device=device)
            else:
                col_i = cols[i]
                d2_i = (col_i - col_i.T).pow(2)
                nz_i = d2_i[d2_i > 0]
                sigma_i = torch.sqrt(0.5 * nz_i.median() + 1e-7) if nz_i.numel() else torch.tensor(1.0, device=device)
                Ki = torch.exp(-d2_i / (2 * sigma_i ** 2))
                _center_kernel_inplace(Ki)
            Ki_list.append(Ki)

        # Inner loop over j >= i_start to fill upper triangle; also process subsequent blocks to avoid storing all Ks
        for j_start in range(i_start, d, block_dim):
            j_end = min(d, j_start + block_dim)

            Kj_list = []
            for j in range(j_start, j_end):
                if const_dim[j]:
                    Kj = torch.zeros(n, n, device=device)
                else:
                    col_j = cols[j]
                    d2_j = (col_j - col_j.T).pow(2)
                    nz_j = d2_j[d2_j > 0]
                    sigma_j = torch.sqrt(0.5 * nz_j.median() + 1e-7) if nz_j.numel() else torch.tensor(1.0, device=device)
                    Kj = torch.exp(-d2_j / (2 * sigma_j ** 2))
                    _center_kernel_inplace(Kj)
                Kj_list.append(Kj)

            # Compute block Frobenius inner products
            for bi, i in enumerate(range(i_start, i_end)):
                for bj, j in enumerate(range(j_start, j_end)):
                    if j <= i:
                        continue
                    val = (Ki_list[bi] * Kj_list[bj]).sum() / norm
                    hsic[i, j] = hsic[j, i] = val

            # Free Kj_list promptly
            del Kj_list

        # Free Ki_list promptly
        del Ki_list

    # 3. global score (mean off-diagonal, ignore zeros from constant dims)
    mask = hsic != 0
    global_score = hsic[mask].mean().item() if mask.any() else 0.0
    
    
    # Create improved HSIC heatmap
    plt.figure(figsize=(10, 8))
    hsic_matrix = hsic.cpu().numpy()
    
    # Create heatmap with better styling
    ax = sns.heatmap(hsic_matrix, 
                     vmin=0, vmax=0.05, 
                     square=True, 
                     cmap="viridis",
                     annot=True if hsic_matrix.shape[0] <= 20 else False,
                     fmt='.3f' if hsic_matrix.shape[0] <= 20 else None,
                     cbar_kws={'label': 'HSIC Score', 'shrink': 0.8},
                     linewidths=0.5,
                     linecolor='white')
    
    sample_text = f" (sampled {n:,}/{original_n:,})" if original_n > n else f" ({n:,} samples)"
    plt.title(f'HSIC Independence Matrix{sample_text}\n(Global Score: {global_score:.4f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Latent Dimension', fontsize=12, fontweight='bold')
    
    # Add text annotation for interpretation
    plt.figtext(0.02, 0.02, 
               'Lower values indicate more independent dimensions\n'
               'Diagonal values are always 0 (self-independence)', 
               fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "hsic_matrix.png"), dpi=300, bbox_inches='tight')
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


def _trustworthiness(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10, max_samples: int = 5000) -> float:
    """Sklearn-like trustworthiness without dependency. Returns [0,1]."""
    n = X_high.shape[0]
    
    # Subsample for computational efficiency if dataset is large
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, size=max_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
        n = max_samples
    
    # Rank distances in original space
    D_X = pairwise_distances(X_high)
    np.fill_diagonal(D_X, np.inf)
    ranks = D_X.argsort(axis=1)
    # Neighbours in low-d space
    D_Y = pairwise_distances(X_low)
    np.fill_diagonal(D_Y, np.inf)
    neigh_low = D_Y.argsort(axis=1)[:, :n_neighbors]
    k = n_neighbors
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


def _continuity(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10, max_samples: int = 5000) -> float:
    """Continuity measure [0,1]."""
    n = X_high.shape[0]
    
    # Subsample for computational efficiency if dataset is large
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, size=max_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
        n = max_samples
    
    D_X = pairwise_distances(X_high)
    np.fill_diagonal(D_X, np.inf)
    neigh_high = D_X.argsort(axis=1)[:, :n_neighbors]
    D_Y = pairwise_distances(X_low)
    np.fill_diagonal(D_Y, np.inf)
    ranks_low = D_Y.argsort(axis=1)
    k = n_neighbors
    c_sum = 0.0
    for i in range(n):
        for j in neigh_high[i]:
            r_ij = int(np.where(ranks_low[i] == j)[0][0]) + 1
            if r_ij > k:
                c_sum += r_ij - k
    denom = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / denom) * c_sum if denom > 0 else 0.0


def _distance_correlation(X_high: np.ndarray, X_low: np.ndarray, max_samples: int = 5000) -> float:
    """Pearson correlation of vectorised pairwise distances (global geometry)."""
    n = X_high.shape[0]
    
    # Subsample for computational efficiency if dataset is large
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, size=max_samples, replace=False)
        X_high = X_high[idx]
        X_low = X_low[idx]
    
    Dh = pairwise_distances(X_high)
    Dl = pairwise_distances(X_low)
    iu = np.triu_indices_from(Dh, k=1)
    a, b = Dh[iu], Dl[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_latent_features(t_latent_features, e_latent_features, results_path: str | os.PathLike[str], 
                            subsample_config: Dict[str, int] = None) -> Dict[str, Any]:
    # Memory monitoring helper
    def _log_memory_usage(stage: str):
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"  Memory usage at {stage}: {memory_mb:.1f} MB")
        except ImportError:
            pass  # psutil not available
    
    """Compute a practical subset of latent-evaluation metrics that do not require labels or original inputs.

    Implemented metrics:
      - active_units: number of dimensions with variance above a small threshold
      - variance_per_dim: per-dimension variance summary
      - hsic_matrix/global_score: independence via HSIC (saved heatmaps per split)
      - cluster quality: silhouette, Davies–Bouldin, Calinski–Harabasz on KMeans(k=5)
      - pca_explained_variance: PCA explained variance ratio on concatenated latents
      
    Args:
        subsample_config: Dict with max sample sizes for different operations:
            - 'hsic': max samples for HSIC calculation (default: 10000)
            - 'geometry': max samples for geometry metrics (default: 5000)
            - 'tsne': max samples for t-SNE (default: 2000)
            - 'shepard': max samples for Shepard plots (default: 10000)
            Setting to None uses full dataset (may be slow for large datasets).
            
    Note on subsampling validity:
        - Random subsampling preserves statistical properties for visualization and most metrics
        - HSIC independence: 10k samples typically sufficient for latent dimension analysis
        - Geometry metrics: 5k samples adequate for neighborhood structure assessment
        - Shepard plots: 10k samples adequate for distance preservation analysis
        - For critical analysis, consider using larger subsets or full dataset
        - All subsampling uses seeded random selection for reproducibility
    """
    # Set default subsampling configuration
    if subsample_config is None:
        subsample_config = {
            'hsic': 10000,
            'geometry': 5000, 
            'tsne': 2000,
            'shepard': 10000  # Limit for Shepard plot pairwise distances
        }
    #convert to numpy arrays, but only the first element of each sample
    train_latent_features = np.array([sample[0] for sample in t_latent_features.dataset])
    eval_latent_features = np.array([sample[0] for sample in e_latent_features.dataset])
    
    z_tr = _to_tensor_2d(train_latent_features)
    z_ev = _to_tensor_2d(eval_latent_features)
    
    # Extract sample IDs if available to check for data integrity
    train_sample_ids = getattr(t_latent_features, "sample_ids", None)
    eval_sample_ids = getattr(e_latent_features, "sample_ids", None)
    
    if train_sample_ids is None or eval_sample_ids is None:
        raise ValueError("Sample IDs are not available. Please ensure the latent features are loaded with sample IDs.")
    
    # Check for overlapping samples between train and eval (potential data leakage)
    if train_sample_ids is not None and eval_sample_ids is not None:
        train_set = set(train_sample_ids)
        eval_set = set(eval_sample_ids)
        overlap = train_set.intersection(eval_set)
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping sample IDs between train and eval sets.")
            print(f"   This may indicate data leakage. Overlapping IDs: {list(overlap)[:5]}{'...' if len(overlap) > 5 else ''}")
        else:
            print(f"✓ No sample overlap detected between train ({len(train_set)}) and eval ({len(eval_set)}) sets.")


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

    print(f"Computing HSIC independence metrics...")
    _log_memory_usage("before HSIC")
    hsic_max = subsample_config.get('hsic', None)  # None means no subsampling
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    hsic_tr = independence_of_features(z_tr, save_path=str(train_dir), device=auto_device, 
                                      max_samples=hsic_max if hsic_max else z_tr.shape[0])
    hsic_ev = independence_of_features(z_ev, save_path=str(eval_dir), device=auto_device, 
                                      max_samples=hsic_max if hsic_max else z_ev.shape[0])
    
    # Force garbage collection after memory-intensive HSIC computations
    import gc
    gc.collect()
    _log_memory_usage("after HSIC cleanup")

    # Cluster metrics (unsupervised)
    print(f"Computing clustering metrics...")
    clus_tr = _cluster_metrics(z_tr.cpu().numpy(), n_clusters=5)
    clus_ev = _cluster_metrics(z_ev.cpu().numpy(), n_clusters=5)

    # PCA explained variance (fit on train, evaluate on both splits to avoid data leakage)
    print(f"Computing PCA analysis...")
    try:
        z_tr_np = z_tr.cpu().numpy()
        z_ev_np = z_ev.cpu().numpy()
        
        # Fit PCA only on training data to avoid data leakage
        pca = PCA(n_components=min(z_tr_np.shape[0], z_tr_np.shape[1]))
        pca.fit(z_tr_np)
        
        # Get explained variance ratio from training fit
        evr = pca.explained_variance_ratio_.astype(np.float32)
        
        # Transform both datasets using the train-fitted PCA
        z_tr_pca = pca.transform(z_tr_np)
        z_ev_pca = pca.transform(z_ev_np)
        # Summaries with train/eval comparison
        pca_summary = {
            "top5_ratio_sum": float(evr[:5].sum()) if evr.size >= 5 else float(evr.sum()),
            "explained_variance_ratio": evr,  # full vector (from train data)
            "train_shape": z_tr_np.shape,
            "eval_shape": z_ev_np.shape,
            "note": "PCA fitted on training data only to avoid data leakage"
        }
        
        # Clear PCA transformation results to free memory
        del z_tr_pca, z_ev_pca

        # Create more meaningful latent space analysis plots
        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
            
            # Original PCA variance plot (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Cumulative explained variance (traditional PCA analysis)
            components = np.arange(1, len(evr) + 1)
            cumsum_evr = np.cumsum(evr)
            
            ax1.plot(components, cumsum_evr, marker="o", linewidth=2.5, 
                    markersize=6, color='#2E86AB', markerfacecolor='#A23B72')
            ax1.fill_between(components, cumsum_evr, alpha=0.3, color='#2E86AB')
            ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
            ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
            
            ax1.set_xlabel("Number of Linear Components", fontweight='bold')
            ax1.set_ylabel("Linear Variance Captured", fontweight='bold')
            ax1.set_title("Linear Dimensionality Analysis\n(May not reflect semantic importance)", fontweight='bold', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05)
            
            # Add warning about interpretation
            ax1.text(0.02, 0.5, "WARNING: High linear variance ≠ semantic importance\nLatent features may use nonlinear structure", 
                    transform=ax1.transAxes, fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Add annotation for 95% variance
            try:
                idx_95 = np.argmax(cumsum_evr >= 0.95)
                if cumsum_evr[idx_95] >= 0.95:
                    ax1.annotate(f'{idx_95 + 1} components\nfor 95% variance', 
                               xy=(idx_95 + 1, cumsum_evr[idx_95]), 
                               xytext=(idx_95 + 1 + len(evr)*0.2, 0.7),
                               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                               fontsize=9, ha='center')
            except:
                pass
            
            # More meaningful latent space analysis (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Latent dimension utilization analysis
            var_tr_sorted = np.sort(var_tr)[::-1]  # Sort descending
            var_ev_sorted = np.sort(var_ev)[::-1]
            
            n_show = min(50, len(var_tr))
            dims = np.arange(1, n_show + 1)
            
            ax2.plot(dims, var_tr_sorted[:n_show], 'o-', label='Training', color='#2E86AB', alpha=0.8)
            ax2.plot(dims, var_ev_sorted[:n_show], 'o-', label='Evaluation', color='#C73E1D', alpha=0.8)
            ax2.axhline(y=1e-3, color='purple', linestyle='--', alpha=0.7, label='Active threshold')
            
            ax2.set_xlabel("Latent Dimension (sorted by variance)", fontweight='bold')
            ax2.set_ylabel("Variance per Dimension", fontweight='bold')
            ax2.set_title("Latent Dimension Utilization\n(More meaningful for neural representations)", fontweight='bold', fontsize=11)
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Effective dimensionality analysis (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Compute effective dimensionality using different thresholds
            thresholds = np.logspace(-6, -1, 50)
            n_active_tr = [(var_tr > thresh).sum() for thresh in thresholds]
            n_active_ev = [(var_ev > thresh).sum() for thresh in thresholds]
            
            ax3.semilogx(thresholds, n_active_tr, 'o-', label='Training', color='#2E86AB')
            ax3.semilogx(thresholds, n_active_ev, 'o-', label='Evaluation', color='#C73E1D')
            ax3.axvline(x=1e-3, color='purple', linestyle='--', alpha=0.7, label='Standard threshold')
            
            ax3.set_xlabel("Variance Threshold", fontweight='bold')
            ax3.set_ylabel("Number of Active Dimensions", fontweight='bold')
            ax3.set_title("Effective Dimensionality vs Threshold\n(Train vs Eval comparison)", fontweight='bold', fontsize=11)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Train vs Eval latent space comparison (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Scatter plot of train vs eval variances per dimension
            max_var = max(var_tr.max(), var_ev.max())
            min_var = min(var_tr.min(), var_ev.min())
            
            # Color points by whether they're "active" in both splits
            colors = []
            for vtr, vev in zip(var_tr, var_ev):
                if vtr > 1e-3 and vev > 1e-3:
                    colors.append('#4CAF50')  # Green: active in both
                elif vtr > 1e-3 or vev > 1e-3:
                    colors.append('#FF9800')  # Orange: active in one
                else:
                    colors.append('#F44336')  # Red: inactive in both
            
            ax4.scatter(var_tr, var_ev, c=colors, alpha=0.7, s=30)
            ax4.plot([min_var, max_var], [min_var, max_var], 'k--', alpha=0.5, label='Perfect agreement')
            
            ax4.set_xlabel("Training Variance", fontweight='bold')
            ax4.set_ylabel("Evaluation Variance", fontweight='bold')
            ax4.set_title("Train vs Eval Variance per Dimension\n(Generalization analysis)", fontweight='bold', fontsize=11)
            ax4.set_xscale('log')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add custom legend for colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4CAF50', label='Active in both'),
                Patch(facecolor='#FF9800', label='Active in one only'), 
                Patch(facecolor='#F44336', label='Inactive in both')
            ]
            ax4.legend(handles=legend_elements, loc='upper left', fontsize=9)
            
            # Summary statistics (bottom panel)
            ax5 = fig.add_subplot(gs[2, :])
            ax5.axis('off')
            
            # Calculate meaningful summary statistics
            active_both = ((var_tr > 1e-3) & (var_ev > 1e-3)).sum()
            active_train_only = ((var_tr > 1e-3) & (var_ev <= 1e-3)).sum()
            active_eval_only = ((var_tr <= 1e-3) & (var_ev > 1e-3)).sum()
            inactive_both = ((var_tr <= 1e-3) & (var_ev <= 1e-3)).sum()
            
            correlation = np.corrcoef(np.log(var_tr + 1e-10), np.log(var_ev + 1e-10))[0, 1]
            
            summary_text = f"""
            LATENT SPACE ANALYSIS SUMMARY:
            
            Dimension Utilization:
            • Active in both train & eval: {active_both}/{len(var_tr)} ({100*active_both/len(var_tr):.1f}%)
            • Active only in training: {active_train_only} (potential overfitting)
            • Active only in evaluation: {active_eval_only} (unusual)
            • Inactive in both: {inactive_both} (unused capacity)
            
            Generalization Analysis:
            • Train-eval variance correlation: {correlation:.3f}
            • Mean variance ratio (eval/train): {np.mean(var_ev)/np.mean(var_tr):.3f}
            
            What This Means:
            • High correlation (>0.8): Good generalization of latent structure
            • Many train-only active dims: Possible overfitting
            • Low overall dimensionality: Efficient representation
            """
            
            ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Remove tight_layout() to avoid gridspec conflicts
            plt.savefig(os.path.join(results_dir, "latent_space_analysis.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception:
            pass
    except Exception:
        pca_summary = {}

    # Geometry preservation metrics with a 2D PCA projection (deterministic)
    try:
        # Fit 2D PCA on training data only, then transform both splits
        pca2 = PCA(n_components=2)
        z_tr_np = z_tr.cpu().numpy()
        z_ev_np = z_ev.cpu().numpy()
        
        pca2.fit(z_tr_np)  # Fit only on training data
        z_tr_2d = pca2.transform(z_tr_np)
        z_ev_2d = pca2.transform(z_ev_np)
        
        # Use subsampling for large datasets to make geometry calculations feasible
        geom_max_samples = subsample_config.get('geometry', None)
        if geom_max_samples:
            print(f"Computing geometry metrics (subsampling to {geom_max_samples:,} if needed)")
        else:
            print(f"Computing geometry metrics (using full dataset)")
            geom_max_samples = max(z_tr.shape[0], z_ev.shape[0])
        
        geom_tr = {
            "trustworthiness": _trustworthiness(z_tr_np, z_tr_2d, n_neighbors=10, 
                                              max_samples=geom_max_samples),
            "continuity": _continuity(z_tr_np, z_tr_2d, n_neighbors=10, 
                                    max_samples=geom_max_samples),
            "dist_corr": _distance_correlation(z_tr_np, z_tr_2d, 
                                             max_samples=geom_max_samples),
        }
        geom_ev = {
            "trustworthiness": _trustworthiness(z_ev_np, z_ev_2d, n_neighbors=10, 
                                              max_samples=geom_max_samples),
            "continuity": _continuity(z_ev_np, z_ev_2d, n_neighbors=10, 
                                    max_samples=geom_max_samples),
            "dist_corr": _distance_correlation(z_ev_np, z_ev_2d, 
                                             max_samples=geom_max_samples),
        }
        # Save improved scatter plots
        def _create_improved_scatter(data_2d, title_suffix, save_path, geometry_metrics=None):
            plt.figure(figsize=(10, 8))
            
            # Create density-based coloring
            from scipy.stats import gaussian_kde
            try:
                xy = data_2d.T
                density = gaussian_kde(xy)(xy)
                idx = density.argsort()
                x_sorted, y_sorted, density_sorted = data_2d[idx, 0], data_2d[idx, 1], density[idx]
            except:
                x_sorted, y_sorted = data_2d[:, 0], data_2d[:, 1]
                density_sorted = None
            
            if density_sorted is not None:
                scatter = plt.scatter(x_sorted, y_sorted, c=density_sorted, s=25, alpha=0.7, 
                                    cmap='viridis', edgecolors='white', linewidth=0.1)
                plt.colorbar(scatter, label='Density', shrink=0.8)
            else:
                plt.scatter(data_2d[:, 0], data_2d[:, 1], s=25, alpha=0.7, 
                          color='#2E86AB', edgecolors='white', linewidth=0.1)
            
            plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} variance)", 
                      fontweight='bold', fontsize=12)
            plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} variance)", 
                      fontweight='bold', fontsize=12)
            plt.title(f"PCA 2D Projection – {title_suffix}\n"
                     f"Total variance explained: {pca2.explained_variance_ratio_.sum():.1%}", 
                     fontweight='bold', fontsize=13)
            
            # Add metrics text if available
            if geometry_metrics:
                metrics_text = (f"Trustworthiness: {geometry_metrics.get('trustworthiness', 0):.3f}\n"
                              f"Continuity: {geometry_metrics.get('continuity', 0):.3f}\n"
                              f"Distance Correlation: {geometry_metrics.get('dist_corr', 0):.3f}")
                plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        _create_improved_scatter(z_tr_2d, "Training", 
                               os.path.join(train_dir, "pca2_scatter.png"), geom_tr)
        _create_improved_scatter(z_ev_2d, "Evaluation", 
                               os.path.join(eval_dir, "pca2_scatter.png"), geom_ev)
        
        # Clear 2D PCA results to free memory before Shepard plots
        del z_tr_2d, z_ev_2d
        
        # Force garbage collection after geometry metrics
        gc.collect()

        # Improved t-SNE (subsample if needed)
        def _tsne_scatter(Z: np.ndarray, out_path: str, title_suffix: str):
            try:
                n = Z.shape[0]
                original_n = n
                tsne_max = subsample_config.get('tsne', 2000)
                if tsne_max and n > tsne_max:
                    idx = np.random.RandomState(42).choice(n, size=tsne_max, replace=False)
                    Z = Z[idx]
                    n = tsne_max
                
                perplexity = max(5, min(30, (n - 1) // 3))
                tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", 
                           learning_rate="auto", random_state=42, n_iter=1000)
                Z2 = tsne.fit_transform(Z)
                
                plt.figure(figsize=(10, 8))
                
                # Create density-based coloring for t-SNE too
                try:
                    from scipy.stats import gaussian_kde
                    xy = Z2.T
                    density = gaussian_kde(xy)(xy)
                    idx_density = density.argsort()
                    x_sorted, y_sorted, density_sorted = Z2[idx_density, 0], Z2[idx_density, 1], density[idx_density]
                    
                    scatter = plt.scatter(x_sorted, y_sorted, c=density_sorted, s=25, alpha=0.8, 
                                        cmap='plasma', edgecolors='white', linewidth=0.1)
                    plt.colorbar(scatter, label='Density', shrink=0.8)
                except:
                    plt.scatter(Z2[:, 0], Z2[:, 1], s=25, alpha=0.8, 
                              color='#F18F01', edgecolors='white', linewidth=0.1)
                
                sample_text = f" (sampled {n:,}/{original_n:,})" if original_n > n else ""
                plt.title(f"t-SNE 2D Projection – {title_suffix}{sample_text}\n"
                         f"Perplexity: {perplexity}", 
                         fontweight='bold', fontsize=13)
                plt.xlabel("t-SNE Dimension 1", fontweight='bold', fontsize=12)
                plt.ylabel("t-SNE Dimension 2", fontweight='bold', fontsize=12)
                
                # Add interpretation note
                plt.figtext(0.02, 0.02, 
                           't-SNE preserves local neighborhood structure\n'
                           'Clusters indicate similar latent representations', 
                           fontsize=8, style='italic', alpha=0.7)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass

        print(f"Computing t-SNE projections...")
        _tsne_scatter(z_tr_np, os.path.join(train_dir, "tsne_scatter.png"), "Training")
        _tsne_scatter(z_ev_np, os.path.join(eval_dir, "tsne_scatter.png"), "Evaluation")

        # Improved Shepard plot: pairwise distances high-d vs 2D
        def _shepard(Z_hd: np.ndarray, Z_2d: np.ndarray, out_path: str, title_suffix: str, max_pairs: int = None):
            try:
                # Use configurable subsampling for memory efficiency
                if max_pairs is None:
                    max_pairs = 10000  # Default fallback
                
                # Subsample data before computing pairwise distances to save memory
                n_samples = Z_hd.shape[0]
                if n_samples > max_pairs:
                    print(f"  Shepard: Subsampling to {max_pairs:,} samples for memory efficiency")
                    idx = np.random.RandomState(42).choice(n_samples, size=max_pairs, replace=False)
                    Z_hd_sub = Z_hd[idx]
                    Z_2d_sub = Z_2d[idx]
                else:
                    Z_hd_sub = Z_hd
                    Z_2d_sub = Z_2d
                
                # Compute pairwise distances on subsampled data
                print(f"    Computing pairwise distances for {Z_hd_sub.shape[0]:,} samples...")
                Dh = pairwise_distances(Z_hd_sub)
                Dl = pairwise_distances(Z_2d_sub)
                
                # Extract upper triangle indices and values
                iu = np.triu_indices_from(Dh, k=1)
                a, b = Dh[iu], Dl[iu]
                m = a.shape[0]
                original_m = m
                
                # Clear distance matrices immediately to free memory
                del Dh, Dl
                
                # Additional safety check for very large datasets
                if m > 50000:  # Hard limit to prevent memory issues
                    idx = np.random.RandomState(42).choice(m, size=50000, replace=False)
                    a = a[idx]
                    b = b[idx]
                    m = 50000
                
                plt.figure(figsize=(10, 8))
                
                # Create density-based coloring for scatter points
                try:
                    plt.hexbin(a, b, gridsize=50, cmap='Blues', alpha=0.8, mincnt=1)
                    plt.colorbar(label='Point Density', shrink=0.8)
                except:
                    plt.scatter(a, b, s=3, alpha=0.4, color='#2E86AB')
                
                # Add trend line
                try:
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(a, b)
                    line_x = np.array([a.min(), a.max()])
                    line_y = slope * line_x + intercept
                    plt.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.8, 
                           label=f'Linear fit (R² = {r_value**2:.3f})')
                    
                    # Add ideal y=x line
                    min_val, max_val = min(a.min(), b.min()), max(a.max(), b.max())
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', 
                           linewidth=1.5, alpha=0.7, label='Perfect preservation (y=x)')
                    
                    plt.legend()
                except Exception:
                    pass
                
                plt.xlabel("Distances in Original Latent Space", fontweight='bold', fontsize=12)
                plt.ylabel("Distances in 2D Projection", fontweight='bold', fontsize=12)
                
                sample_text = f" ({m:,}/{original_m:,} pairs)" if original_m > m else f" ({m:,} pairs)"
                plt.title(f"Shepard Plot – {title_suffix}{sample_text}\n"
                         "Distance Preservation Analysis", 
                         fontweight='bold', fontsize=13)
                
                # Add interpretation note
                plt.figtext(0.02, 0.02, 
                           'Points near y=x line indicate good distance preservation\n'
                           'Scatter indicates distortion in the embedding', 
                           fontsize=8, style='italic', alpha=0.7)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception:
                pass

        print(f"Computing Shepard plots...")
        _log_memory_usage("before Shepard plots")
        shepard_max = subsample_config.get('shepard', 10000)
        
        # Recreate 2D projections just for Shepard plots (memory efficient)
        pca2_shepard = PCA(n_components=2)
        pca2_shepard.fit(z_tr_np)  # Fit on training data only
        z_tr_2d_shepard = pca2_shepard.transform(z_tr_np)
        z_ev_2d_shepard = pca2_shepard.transform(z_ev_np)
        
        _shepard(z_tr_np, z_tr_2d_shepard, os.path.join(train_dir, "shepard_plot.png"), "Training", shepard_max)
        _shepard(z_ev_np, z_ev_2d_shepard, os.path.join(eval_dir, "shepard_plot.png"), "Evaluation", shepard_max)
        
        # Clear Shepard-specific arrays
        del z_tr_2d_shepard, z_ev_2d_shepard, pca2_shepard
        
        # Final memory cleanup after all heavy computations
        gc.collect()
        _log_memory_usage("after Shepard plots cleanup")
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

    # Improved distribution plots: variance per dim
    def _create_variance_plot(variance_data, title_suffix, save_path, split_metrics):
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplot layout
            gs = plt.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1])
            
            # Main histogram
            ax1 = plt.subplot(gs[0, 0])
            n_bins = min(50, max(10, len(variance_data) // 10))
            counts, bins, patches = ax1.hist(variance_data, bins=n_bins, alpha=0.7, 
                                           color='#2E86AB', edgecolor='black', linewidth=0.5)
            
            # Color bars based on variance magnitude
            for i, (count, patch) in enumerate(zip(counts, patches)):
                if bins[i] < 1e-6:
                    patch.set_facecolor('#FF6B6B')  # Red for very low variance (inactive)
                elif bins[i] < 1e-3:
                    patch.set_facecolor('#FFE66D')  # Yellow for low variance
                else:
                    patch.set_facecolor('#4ECDC4')  # Teal for active variance
            
            # Add statistics
            mean_var = np.mean(variance_data)
            median_var = np.median(variance_data)
            std_var = np.std(variance_data)
            
            ax1.axvline(mean_var, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_var:.4f}')
            ax1.axvline(median_var, color='orange', linestyle='--', linewidth=2, 
                       label=f'Median: {median_var:.4f}')
            ax1.axvline(1e-3, color='purple', linestyle=':', linewidth=2, 
                       label='Active threshold (1e-3)')
            
            ax1.set_xlabel("Variance per Latent Dimension", fontweight='bold', fontsize=12)
            ax1.set_ylabel("Number of Dimensions", fontweight='bold', fontsize=12)
            ax1.set_title(f"Latent Dimension Variance Distribution – {title_suffix}\n"
                         f"Active Dimensions: {split_metrics.get('active_units', 0)}/{len(variance_data)}", 
                         fontweight='bold', fontsize=13)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log') if max(counts) > 100 else None
            
            # Box plot
            ax2 = plt.subplot(gs[0, 1])
            box_plot = ax2.boxplot(variance_data, vert=True, patch_artist=True, 
                                  boxprops=dict(facecolor='#A8E6CF', alpha=0.7))
            ax2.set_ylabel("Variance", fontweight='bold')
            ax2.set_title("Distribution\nSummary", fontweight='bold', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # Statistics table
            ax3 = plt.subplot(gs[1, :])
            ax3.axis('off')
            
            stats_data = [
                ['Statistic', 'Value'],
                ['Mean', f'{mean_var:.2e}'],
                ['Median', f'{median_var:.2e}'],
                ['Std Dev', f'{std_var:.2e}'],
                ['Min', f'{np.min(variance_data):.2e}'],
                ['Max', f'{np.max(variance_data):.2e}'],
                ['Active Dims (>1e-3)', f'{(variance_data > 1e-3).sum()}'],
                ['Near-zero Dims (<1e-6)', f'{(variance_data < 1e-6).sum()}']
            ]
            
            table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                            cellLoc='center', loc='center', 
                            colWidths=[0.3, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Style the table
            for i in range(len(stats_data)):
                for j in range(len(stats_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4ECDC4')
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('#F0F0F0')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            # Fallback to simple histogram
            try:
                plt.figure(figsize=(8, 6))
                plt.hist(variance_data, bins=30, alpha=0.7, color='#2E86AB')
                plt.xlabel(f"Variance per latent dimension ({title_suffix.lower()})", fontweight='bold')
                plt.ylabel("Count", fontweight='bold')
                plt.title(f"Variance Distribution – {title_suffix}", fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
    
    print(f"Creating variance distribution plots...")
    try:
        _create_variance_plot(var_tr, "Training", 
                            os.path.join(train_dir, "variance_hist.png"), 
                            metrics['train'])
        _create_variance_plot(var_ev, "Evaluation", 
                            os.path.join(eval_dir, "variance_hist.png"), 
                            metrics['eval'])
    except Exception:
        pass

    print(f"Computing mutual information metrics...")
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

    # Add train vs eval comparison metrics
    try:
        comparison_metrics = {
            "active_units_ratio": metrics["eval"]["active_units"] / max(metrics["train"]["active_units"], 1),
            "hsic_score_ratio": metrics["eval"]["hsic_global_score"] / max(metrics["train"]["hsic_global_score"], 1e-10),
            "variance_distribution": {
                "train_mean": float(np.mean(var_tr)),
                "eval_mean": float(np.mean(var_ev)),
                "ratio": float(np.mean(var_ev)) / max(float(np.mean(var_tr)), 1e-10)
            }
        }
        
        # Add geometry comparison if available
        if geom_tr and geom_ev:
            comparison_metrics["geometry_preservation"] = {
                "trustworthiness_ratio": geom_ev.get("trustworthiness", 0) / max(geom_tr.get("trustworthiness", 1e-10), 1e-10),
                "continuity_ratio": geom_ev.get("continuity", 0) / max(geom_tr.get("continuity", 1e-10), 1e-10)
            }
        
        metrics["train_vs_eval_comparison"] = comparison_metrics
    except Exception:
        pass

    # Add subsampling information to metrics for transparency
    metrics["subsampling_info"] = {
        "hsic_max_samples": subsample_config.get('hsic'),
        "geometry_max_samples": subsample_config.get('geometry'), 
        "tsne_max_samples": subsample_config.get('tsne'),
        "shepard_max_samples": subsample_config.get('shepard'),
        "note": "None means no subsampling was applied. Subsampling uses seeded random sampling for reproducibility."
    }
    
    # Add recommendations for large dataset analysis
    n_total = metrics["train"]["n_samples"] + metrics["eval"]["n_samples"]
    if n_total > 20000:
        recommendations = {
            "dataset_size": "large",
            "most_useful_plots": [
                "HSIC heatmap: Shows latent independence patterns",
                "Latent space analysis: Reveals dimension utilization and generalization", 
                "Variance histograms: Identifies active vs inactive dimensions",
                "Train vs eval scatter: Detects overfitting in latent space"
            ],
            "less_useful_at_scale": [
                "Individual scatter points: Too dense to interpret",
                "t-SNE: Representative but subsampled",
                "Shepard plots: Shows overall trends but subsampled"
            ],
            "key_metrics_to_focus_on": [
                "Active units ratio (train vs eval)",
                "HSIC global score", 
                "Dimension utilization patterns",
                "Train-eval variance correlation",
                "Effective dimensionality"
            ]
        }
        metrics["analysis_recommendations"] = recommendations
    
    print(f"✅ Latent feature evaluation completed!")
    print(f"Dataset size: {n_total:,} total samples")
    if n_total > 20000:
        print(f"For large datasets, focus on: HSIC heatmap, latent space analysis, variance distributions, and train/eval comparisons")
        print(f"The new latent space analysis plot shows dimension utilization and generalization patterns")
    
    return metrics
