#!/usr/bin/env python3
"""
Comprehensive metrics comparison and visualization script for EEG feature extraction methods.
This script loads metrics from different extraction methods and creates comparative plots and tables.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from math import pi
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import shutil
#arnings.filterwarnings('ignore')

# Import pairwise comparison functions
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
import seaborn as sns

# Try to import CUDA libraries for GPU acceleration
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    CUDA_AVAILABLE = False

# =================================================================
# PAIRWISE SIMILARITY ACCELERATION SETTINGS
# =================================================================
# Choose your preferred acceleration method:

# Option 1: GPU Acceleration (fastest for large datasets)
PAIRWISE_USE_GPU = False  # Set to True to use CUDA GPU

# Option 2: CPU Multiprocessing (great for 16-core systems)  
PAIRWISE_N_WORKERS = 32 # Conservative setting (1 = sequential)

# Option 3: Automatic (tries GPU first, then multicore, then sequential)
PAIRWISE_AUTO = False    # Set to False to use manual settings above

# Option 4: Include CCA metric (can be slow/non-convergent)
PAIRWISE_INCLUDE_CCA = True

# Set scientific style for plots
plt.style.use('default')  # Use clean default style
sns.set_style("whitegrid")  # Clean grid background
sns.set_palette("Set2")  # Professional color palette

# Set consistent scientific styling
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,          # +50%
    'axes.titlesize': 21,     # +50%
    'axes.labelsize': 18,     # +50%
    'xtick.labelsize': 15,    # +50%
    'ytick.labelsize': 15,    # +50%
    'legend.fontsize': 15,    # +50%
    'figure.titlesize': 24,   # +50%
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False  # Set to True if LaTeX is available
})

# =================================================================
# GLOBAL WORKER FUNCTION FOR MULTIPROCESSING
# =================================================================

def _global_compute_pair_worker(args):
    """Global worker function for multiprocessing (needs to be picklable)."""
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from scipy.spatial import procrustes
    
    pair_info, method_data, k = args
    i, j, method1, method2 = pair_info
    
    # Helper functions (copied from class)
    def _center_rows(X):
        X = np.asarray(X, dtype=np.float64)
        return X - X.mean(axis=0, keepdims=True)

    def _hsic_linear(X, Y):
        Xc = _center_rows(X)
        Yc = _center_rows(Y)
        K = Xc @ Xc.T
        L = Yc @ Yc.T
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        return float(np.sum(KH * LH))

    def linear_cka(Z1, Z2):
        hsic_xy = _hsic_linear(Z1, Z2)
        hsic_xx = _hsic_linear(Z1, Z1)
        hsic_yy = _hsic_linear(Z2, Z2)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        val = hsic_xy / denom
        return float(np.clip(val, -1.0, 1.0))

    def rbf_cka(Z1, Z2):
        def _rbf_kernel(Z, gamma=None):
            D2 = pairwise_distances(Z, metric="sqeuclidean")
            if gamma is None:
                nz = D2[D2 > 0]
                if nz.size == 0:
                    gamma_eff = 1.0
                else:
                    gamma_eff = 1.0 / (2.0 * np.median(nz))
            else:
                gamma_eff = gamma
            return np.exp(-gamma_eff * D2)

        K = _rbf_kernel(Z1)
        L = _rbf_kernel(Z2)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        hsic_xy = float(np.sum(KH * LH))
        hsic_xx = float(np.sum(KH * KH))
        hsic_yy = float(np.sum(LH * LH))
        val = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return float(np.clip(val, -1.0, 1.0))

    def cca_maxcorr(Z1, Z2):
        Z1 = np.asarray(Z1, dtype=np.float64)
        Z2 = np.asarray(Z2, dtype=np.float64)
        n, d1 = Z1.shape
        _, d2 = Z2.shape
        if n < 2 or d1 == 0 or d2 == 0:
            return 0.0
        k_comp = min(d1, d2, n - 1)
        if k_comp < 1:
            return 0.0
        cca = CCA(n_components=k_comp, max_iter=10000)
        Xc = _center_rows(Z1)
        Yc = _center_rows(Z2)
        try:
            Xc, Yc = cca.fit_transform(Xc, Yc)
            corrs = []
            for i in range(Xc.shape[1]):
                x = Xc[:, i]
                y = Yc[:, i]
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    corrs.append(0.0)
                else:
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
            return float(np.max(np.abs(corrs))) if corrs else 0.0
        except:
            return 0.0

    def distance_geometry_corr(Z1, Z2):
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        iu = np.triu_indices_from(D1, k=1)
        a, b = D1[iu], D2[iu]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def knn_jaccard_overlap(Z1, Z2, k):
        n = Z1.shape[0]
        if n <= 1 or k < 1:
            return 0.0
        k = min(k, n - 1)
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        np.fill_diagonal(D1, np.inf)
        np.fill_diagonal(D2, np.inf)
        idx1 = np.argsort(D1, axis=1)[:, :k]
        idx2 = np.argsort(D2, axis=1)[:, :k]
        scores = []
        for i in range(n):
            s1 = set(idx1[i].tolist())
            s2 = set(idx2[i].tolist())
            inter = len(s1 & s2)
            union = len(s1 | s2)
            scores.append(inter / union if union > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    def procrustes_disparity(Z1, Z2):
        try:
            A, B = Z1, Z2
            if A.shape[1] != B.shape[1]:
                k_comp = min(A.shape[1], B.shape[1])
                if k_comp < 1:
                    return 0.0
                p1 = PCA(n_components=k_comp, random_state=42)
                p2 = PCA(n_components=k_comp, random_state=42)
                A = p1.fit_transform(A)
                B = p2.fit_transform(B)
            _, _, disparity = procrustes(A, B)
            return float(disparity)
        except Exception:
            return 0.0

    def _align_latent_features_by_ids(Z1, ids1, Z2, ids2):
        if not ids1 or not ids2:
            raise ValueError("Missing sample IDs for alignment.")
        
        idx1_map = {sid: i for i, sid in enumerate(ids1)}
        idx2_map = {sid: i for i, sid in enumerate(ids2)}
        
        common_ids = [sid for sid in ids1 if sid in idx2_map]
        
        if not common_ids:
            raise ValueError("No overlapping sample IDs between methods; cannot align.")
        
        aligned_idx1 = np.array([idx1_map[sid] for sid in common_ids], dtype=int)
        aligned_idx2 = np.array([idx2_map[sid] for sid in common_ids], dtype=int)
        
        return Z1[aligned_idx1], Z2[aligned_idx2]

    try:
        data1 = method_data[method1]
        data2 = method_data[method2]
        
        Z1, ids1 = data1['features'], data1['ids']
        Z2, ids2 = data2['features'], data2['ids']
        
        if i == j:
            # Same method
            similarities = {
                "cka_linear": 1.0,
                "cka_rbf": 1.0,
                "dist_geom_corr": 1.0,
                f"knn_jaccard_k{k}": 1.0,
                "procrustes_disparity": 0.0,
            }
            if PAIRWISE_INCLUDE_CCA:
                similarities["cca_maxcorr"] = 1.0
            aligned_samples = len(ids1)
        else:
            # Different methods - align and compute
            Z1_aligned, Z2_aligned = _align_latent_features_by_ids(Z1, ids1, Z2, ids2)
            aligned_samples = Z1_aligned.shape[0]
            
            # Compute similarities
            similarities = {
                "cka_linear": linear_cka(Z1_aligned, Z2_aligned),
                "cka_rbf": rbf_cka(Z1_aligned, Z2_aligned),
                "dist_geom_corr": distance_geometry_corr(Z1_aligned, Z2_aligned),
                f"knn_jaccard_k{k}": knn_jaccard_overlap(Z1_aligned, Z2_aligned, k),
                "procrustes_disparity": procrustes_disparity(Z1_aligned, Z2_aligned),
            }
            if PAIRWISE_INCLUDE_CCA:
                similarities["cca_maxcorr"] = cca_maxcorr(Z1_aligned, Z2_aligned)
        
        # Return results with metadata
        results = []
        for metric, value in similarities.items():
            results.append({
                'method1': method1,
                'method2': method2,
                'method1_clean': method1.replace('tuh-', ''),
                'method2_clean': method2.replace('tuh-', ''),
                'metric': metric,
                'value': value,
                'samples_used': aligned_samples
            })
        return results
        
    except Exception as e:
        print(f"    ✗ Error in worker for {method1} vs {method2}: {e}")
        return []
# Font size already set in global config above

class MetricsComparison:
    def __init__(self, results_dir: str = "/rds/general/user/lrh24/home/thesis/code/Results", 
                 method_group: Dict[str, List[str]] = None):
        """Initialize the metrics comparison with the results directory and optional method group.
        
        Args:
            results_dir: Path to the results directory
            method_group: Dict with group name as key and list of methods as value
                        e.g., {"neural_networks": ["tuh-eegnet", "tuh-ctm_nn_pc"]}
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "metrics_and_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define all available methods (ordered by method groups)
        small_aggregated = ["tuh-ctm_cma_avg", "tuh-ctm_nn_avg", "tuh-hopf_avg", "tuh-jr_avg", "tuh-wong_wang_avg", "tuh-pca_avg", "tuh-psd_ae_avg"]
        medium_unrestricted = ["tuh-ctm_nn_pc", "tuh-hopf_pc", "tuh-jr_pc", "tuh-c22", "tuh-pca_pc", "tuh-psd_ae_pc", "tuh-eegnet"]
        self.all_methods = small_aggregated + medium_unrestricted
        
        # Set up method filtering and output directory
        if method_group:
            self.group_name = list(method_group.keys())[0]
            self.methods = method_group[self.group_name]
            # Create subfolder for the group
            self.output_dir = self.output_dir / self.group_name
            self.output_dir.mkdir(exist_ok=True)
            print(f"Analyzing method group: '{self.group_name}'")
            print(f"Methods in group: {[self.clean_method_name(m) for m in self.methods]}")
        else:
            self.group_name = "all_methods"
            self.methods = self.all_methods
            print("Analyzing all available methods")
        
        self.metrics_data = {}
        self.latent_features_cache = {}  # Cache for loaded latent features
        
    def clean_method_name(self, method_name: str) -> str:
        """Clean method name by removing prefixes."""
        # Remove 'tuh-' prefix only
        cleaned = method_name.replace('tuh-', '')
        return cleaned
    
    def get_canonical_method_order(self, methods: list) -> list:
        """Return methods in canonical order, falling back to alphabetical for unknown methods."""
        # Create a mapping from method name to canonical order
        canonical_full = self.all_methods
        canonical_clean = [self.clean_method_name(m) for m in canonical_full]
        
        # Create order mapping
        order_map = {}
        for i, method in enumerate(canonical_clean):
            order_map[method] = i
        
        # Sort input methods by canonical order, unknown methods go to end alphabetically
        def sort_key(method):
            clean_method = self.clean_method_name(method) if method.startswith('tuh-') else method
            return (order_map.get(clean_method, 1000), clean_method)
        
        return sorted(methods, key=sort_key)
    
    def get_method_color(self, method_name: str) -> str:
        """Get color for method name based on method type."""
        # Define method groups
        data_driven_methods = {
            'psd_ae_avg', 'psd_ae_pc', 'c22', 'pca_avg', 'pca_pc', 'eegnet'
        }
        mechanistic_methods = {
            'hopf_avg', 'hopf_pc', 'jr_avg', 'jr_pc', 'wong_wang_avg','ctm_cma_avg', 'ctm_nn_avg', 'ctm_nn_pc'
        }
        
        # Clean the method name
        clean_name = self.clean_method_name(method_name)
        
        # Return appropriate color
        if clean_name in data_driven_methods:
            return '#1F4E79'  # methodblue (data-driven: blue)
        elif clean_name in mechanistic_methods:
            return '#BC3B00'  # methodorange (mechanistic: orange)
        else:
            return '#5B9BD5'  # Muted blue
    
    def get_professional_colors(self):
        """Get a minimal, consistent color palette for plots."""
        return {
            'primary_blue':    '#3566a8',  # Professional blue for main elements
            'primary_purple':  '#6C5B7B',  # Muted purple for contrast
            'secondary_blue':  '#4A90E2',  # Lighter blue for secondary elements
            'secondary_purple':'#8E7CC3',  # Softer purple for secondary
            'accent_green':    '#3CAEA3',  # Teal-green accent
            'accent_orange':   '#F5A623',  # Warm orange accent
            'accent_red':      '#D7263D',  # Professional red accent
            'neutral_gray':    '#7B8A8B',  # Muted gray for neutral elements
            'light_gray':      '#D6DBDF',  # Light gray for highlights
            'dark_gray':       '#2C3E50'   # Dark gray for text/elements
        }
        
    
    def set_colored_xticklabels(self, ax, methods, **kwargs):
        """Set x-tick labels with method-specific colors."""
        ax.set_xticklabels(methods, **kwargs)
        
        # Apply colors to each tick label
        for i, method in enumerate(methods):
            color = self.get_method_color(method)
            ax.get_xticklabels()[i].set_color(color)
            ax.get_xticklabels()[i].set_fontweight('bold')
    
    def set_colored_yticklabels(self, ax, methods, **kwargs):
        """Set y-tick labels with method-specific colors."""
        ax.set_yticklabels(methods, **kwargs)
        
        # Apply colors to each tick label
        for i, method in enumerate(methods):
            color = self.get_method_color(method)
            ax.get_yticklabels()[i].set_color(color)
            ax.get_yticklabels()[i].set_fontweight('bold')
        
    def load_metrics(self) -> Dict[str, Dict]:
        """Load metrics from all available methods."""
        print("Loading metrics from all methods...")
        
        for method in self.methods:
            method_dir = self.results_dir / method
            metrics_file = method_dir / "final_metrics.json"
            
            if metrics_file.exists():
                print(f"  ✓ Loading {method}")
                try:
                    with open(metrics_file, 'r') as f:
                        self.metrics_data[method] = json.load(f)
                except Exception as e:
                    print(f"  ✗ Error loading {method}: {e}")
                    self.metrics_data[method] = None
            else:
                print(f"  ✗ No final_metrics.json for {method}")
                self.metrics_data[method] = None
        
        # Remove None values for actual analysis but keep track of missing methods
        self.available_methods = {k: v for k, v in self.metrics_data.items() if v is not None}
        self.missing_methods = [k for k, v in self.metrics_data.items() if v is None]
        
        print(f"\nSuccessfully loaded {len(self.available_methods)} methods")
        print(f"Missing metrics for {len(self.missing_methods)} methods: {self.missing_methods}")
        
        return self.metrics_data
    
    def extract_classification_metrics(self) -> pd.DataFrame:
        """Extract classification performance metrics for comparison."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'metrics_per_task' not in data:
                continue
                
            for task in ['gender', 'abnormal']:
                if task in data['metrics_per_task']:
                    task_metrics = data['metrics_per_task'][task]
                    row = {
                        'method': method,
                        'task': task,
                        'accuracy': task_metrics.get('accuracy'),
                        'f1': task_metrics.get('f1'),
                        'f1_macro': task_metrics.get('f1_macro'),
                        'precision': task_metrics.get('precision'),
                        'recall': task_metrics.get('recall'),
                        'roc_auc': task_metrics.get('roc_auc'),
                        'pr_auc': task_metrics.get('pr_auc'),
                        'loss': task_metrics.get('loss')
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def extract_latent_metrics(self) -> pd.DataFrame:
        """Extract latent space analysis metrics."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'latent' not in data:
                continue
            
            for split in ['train', 'eval']:
                if split in data['latent']:
                    latent_data = data['latent'][split]
                    variance_per_dim = latent_data.get('variance_per_dim', [])
                    
                    row = {
                        'method': method,
                        'split': split,
                        'dim': latent_data.get('dim'),
                        'active_units': latent_data.get('active_units'),
                        'hsic_global_score': latent_data.get('hsic_global_score'),
                        'silhouette': latent_data.get('cluster', {}).get('silhouette'),
                        'davies_bouldin': latent_data.get('cluster', {}).get('davies_bouldin'),
                        'calinski_harabasz': latent_data.get('cluster', {}).get('calinski_harabasz'),
                        'trustworthiness': latent_data.get('geometry', {}).get('trustworthiness'),
                        'continuity': latent_data.get('geometry', {}).get('continuity'),
                        'dist_corr': latent_data.get('geometry', {}).get('dist_corr'),
                        'variance_mean': np.mean(variance_per_dim) if variance_per_dim else None,
                        'variance_std': np.std(variance_per_dim) if variance_per_dim else None,
                        'variance_max': np.max(variance_per_dim) if variance_per_dim else None,
                        'variance_min': np.min(variance_per_dim) if variance_per_dim else None,
                        'variance_per_dim': variance_per_dim,  # Store full vector for detailed analysis
                    }
                    
                    # Enhanced variance metrics
                    if variance_per_dim:
                        # Coefficient of variation for variance distribution
                        variance_mean = np.mean(variance_per_dim)
                        if variance_mean > 0:
                            row['variance_cv'] = np.std(variance_per_dim) / variance_mean
                        else:
                            row['variance_cv'] = None
                        
                        # Variance concentration (what fraction of total variance is in top 20% dimensions)
                        sorted_var = np.sort(variance_per_dim)[::-1]
                        top_20_count = max(1, len(sorted_var) // 5)
                        row['variance_concentration'] = np.sum(sorted_var[:top_20_count]) / np.sum(sorted_var)
                        
                        # Effective dimensionality based on variance (entropy-based)
                        var_probs = variance_per_dim / np.sum(variance_per_dim)
                        var_probs = var_probs[var_probs > 0]  # Remove zeros to avoid log(0)
                        if len(var_probs) > 1:
                            row['variance_entropy'] = -np.sum(var_probs * np.log(var_probs))
                            row['effective_dim_variance'] = np.exp(row['variance_entropy'])
                        else:
                            row['variance_entropy'] = None
                            row['effective_dim_variance'] = None
                    
                    # Enhanced mutual information metrics
                    if 'mi_zy' in latent_data:
                        for task in ['gender', 'abnormal', 'age']:  # Include age if available
                            if task in latent_data['mi_zy']:
                                task_mi = latent_data['mi_zy'][task]
                                row[f'mi_{task}_mean'] = task_mi.get('mean')
                                
                                # Per-dimension MI analysis
                                per_dim_mi = task_mi.get('per_dim', [])
                                if per_dim_mi:
                                    row[f'mi_{task}_per_dim'] = per_dim_mi
                                    row[f'mi_{task}_max'] = np.max(per_dim_mi)
                                    row[f'mi_{task}_std'] = np.std(per_dim_mi)
                                    
                                    # MI concentration (similar to variance concentration)
                                    sorted_mi = np.sort(per_dim_mi)[::-1]
                                    top_20_count = max(1, len(sorted_mi) // 5)
                                    row[f'mi_{task}_concentration'] = np.sum(sorted_mi[:top_20_count]) / np.sum(sorted_mi) if np.sum(sorted_mi) > 0 else None
                                    
                                    # Count highly informative dimensions (MI > mean + std)
                                    mi_threshold = np.mean(per_dim_mi) + np.std(per_dim_mi)
                                    row[f'mi_{task}_high_info_dims'] = np.sum(np.array(per_dim_mi) > mi_threshold)
                    
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def extract_pca_metrics(self) -> pd.DataFrame:
        """Extract PCA-related metrics."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'latent' in data and 'pca' in data['latent']:
                pca_data = data['latent']['pca']
                row = {
                    'method': method,
                    'top5_ratio_sum': pca_data.get('top5_ratio_sum'),
                    'explained_variance_top_component': pca_data.get('explained_variance_ratio', [None])[0],
                    'effective_dim_95': None,  # Calculate this
                    'effective_dim_99': None   # Calculate this
                }
                
                # Calculate effective dimensionality
                if 'explained_variance_ratio' in pca_data:
                    cumsum = np.cumsum(pca_data['explained_variance_ratio'])
                    row['effective_dim_95'] = np.argmax(cumsum >= 0.95) + 1
                    row['effective_dim_99'] = np.argmax(cumsum >= 0.99) + 1
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_classification_comparison_plots(self, df: pd.DataFrame):
        """Create individual classification metrics comparison plots with different chart types."""
        # Clean method names in the dataframe
        df = df.copy()
        df['method_clean'] = df['method'].apply(self.clean_method_name)
        
        metrics = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
        tasks = ['gender', 'abnormal']
        
        # Create radar chart for overall performance comparison
        for task in tasks:
            task_data = df[df['task'] == task].dropna(subset=metrics)
            if not task_data.empty:
                self._create_radar_chart(task_data, metrics, task)
        
        # Create individual metric plots with different chart types
        for metric in metrics:
            for task in tasks:
                task_data = df[df['task'] == task].dropna(subset=[metric])
                if not task_data.empty:
                    self._create_metric_plot(task_data, metric, task)
        
        print("  ✓ Saved individual classification comparison plots")
    
    def _create_classification_summary_comparison(self, df: pd.DataFrame):
        """Create classification performance summary comparison plot."""
        # Clean method names in the dataframe
        df = df.copy()
        df['method_clean'] = df['method'].apply(self.clean_method_name)
        
        # Prepare data for comparison
        comparison_data = []
        for method in sorted(df['method_clean'].unique()):
            method_data = {'method': method}
            
            for task in ['gender', 'abnormal']:
                task_data = df[(df['method_clean'] == method) & (df['task'] == task)]
                if not task_data.empty:
                    for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
                        if metric in task_data.columns and not pd.isna(task_data.iloc[0][metric]):
                            method_data[f'{task}_{metric}'] = task_data.iloc[0][metric]
            
            comparison_data.append(method_data)
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Classification Performance: Method Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        methods = comp_df['method']
        x_pos = range(len(methods))
        width = 0.35
        
        # Plot 1: Accuracy Comparison
        if 'gender_accuracy' in comp_df.columns and 'abnormal_accuracy' in comp_df.columns:
            gender_acc = comp_df['gender_accuracy'].fillna(0)
            abnormal_acc = comp_df['abnormal_accuracy'].fillna(0)
            
            colors = self.get_professional_colors()
            bars1a = ax1.bar([x - width/2 for x in x_pos], gender_acc, width,
                           label='Gender Task', alpha=0.8, color=colors['primary_blue'])
            bars1b = ax1.bar([x + width/2 for x in x_pos], abnormal_acc, width,
                           label='Abnormal Task', alpha=0.8, color=colors['primary_purple'])
            
            ax1.set_xticks(x_pos)
            self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right')
            ax1.set_ylabel('Accuracy', fontweight='bold')
            ax1.set_title('A) Classification Accuracy', fontweight='bold', pad=15)
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax1.set_ylim(0, 1.05)
        
        # Plot 2: F1 Score Comparison
        if 'gender_f1' in comp_df.columns and 'abnormal_f1' in comp_df.columns:
            gender_f1 = comp_df['gender_f1'].fillna(0)
            abnormal_f1 = comp_df['abnormal_f1'].fillna(0)
            
            bars2a = ax2.bar([x - width/2 for x in x_pos], gender_f1, width,
                           label='Gender Task', alpha=0.8, color=colors['primary_blue'])
            bars2b = ax2.bar([x + width/2 for x in x_pos], abnormal_f1, width,
                           label='Abnormal Task', alpha=0.8, color=colors['primary_purple'])
            
            ax2.set_xticks(x_pos)
            self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right')
            ax2.set_ylabel('F1 Score', fontweight='bold')
            ax2.set_title('B) F1 Score Performance', fontweight='bold', pad=15)
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax2.set_ylim(0, 1.05)
        
        # Plot 3: ROC AUC Comparison
        if 'gender_roc_auc' in comp_df.columns and 'abnormal_roc_auc' in comp_df.columns:
            gender_auc = comp_df['gender_roc_auc'].fillna(0)
            abnormal_auc = comp_df['abnormal_roc_auc'].fillna(0)
            
            bars3a = ax3.bar([x - width/2 for x in x_pos], gender_auc, width,
                           label='Gender Task', alpha=0.8, color=colors['primary_blue'])
            bars3b = ax3.bar([x + width/2 for x in x_pos], abnormal_auc, width,
                           label='Abnormal Task', alpha=0.8, color=colors['primary_purple'])
            
            ax3.set_xticks(x_pos)
            self.set_colored_xticklabels(ax3, methods, rotation=45, ha='right')
            ax3.set_ylabel('ROC AUC', fontweight='bold')
            ax3.set_title('C) ROC AUC Performance', fontweight='bold', pad=15)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
            ax3.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax3.set_ylim(0, 1.05)
        
        # Plot 4: Average Performance
        if ('gender_accuracy' in comp_df.columns and 'abnormal_accuracy' in comp_df.columns and
            'gender_f1' in comp_df.columns and 'abnormal_f1' in comp_df.columns):
            
            # Calculate average performance across tasks and key metrics
            avg_performance = []
            for _, row in comp_df.iterrows():
                scores = []
                for task in ['gender', 'abnormal']:
                    for metric in ['accuracy', 'f1']:
                        col = f'{task}_{metric}'
                        if col in row and not pd.isna(row[col]):
                            scores.append(row[col])
                
                avg_performance.append(np.mean(scores) if scores else 0)
            
            bars4 = ax4.bar(x_pos, avg_performance, alpha=0.8, color='#5B9BD5')
            ax4.set_xticks(x_pos)
            self.set_colored_xticklabels(ax4, methods, rotation=45, ha='right')
            ax4.set_ylabel('Average Performance', fontweight='bold')
            ax4.set_title('D) Overall Classification Performance\n(Average of Accuracy & F1)', fontweight='bold', pad=15)
            ax4.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax4.set_ylim(0, 1.05)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars4, avg_performance)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Apply consistent scientific styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
            # Make x-axis labels bold
            for tick in ax.get_xticklabels():
                tick.set_fontweight('bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'classification_summary_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        print("  ✓ Saved classification summary comparison plot")
    
    def _create_radar_chart(self, data: pd.DataFrame, metrics: List[str], task: str):
        """Create a radar chart for classification metrics."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for each method
        colors = ['#5B9BD5'] * len(data)
        
        for idx, (_, row) in enumerate(data.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['method_clean'], 
                   color=colors[idx], alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f'{task.title()} Task - Performance Radar Chart', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'radar_chart_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_plot(self, data: pd.DataFrame, metric: str, task: str):
        """Create individual metric plots with different visualization types."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if metric in ['accuracy', 'f1']:
            # Horizontal bar plot for accuracy and F1
            colors = self.get_professional_colors()
            bars = ax.barh(range(len(data)), data[metric], color=colors['primary_blue'], alpha=0.8)
            ax.set_yticks(range(len(data)))
            self.set_colored_yticklabels(ax, data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{task.title()} Task - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Highlight best performance
            best_idx = data[metric].idxmax()
            best_pos = list(data.index).index(best_idx)
            bars[best_pos].set_color('#87CEEB')
            bars[best_pos].set_alpha(1.0)
            
        else:  # roc_auc, pr_auc
            # Lollipop plot for AUC metrics
            sorted_data = data.sort_values(metric)
            y_pos = range(len(sorted_data))
            
            # Create stems
            ax.hlines(y_pos, 0, sorted_data[metric], colors='gray', alpha=0.4, linewidth=2)
            # Create circles
            colors = ['#87CEEB' if val == sorted_data[metric].max() else '#5B9BD5' for val in sorted_data[metric]]
            ax.scatter(sorted_data[metric], y_pos, color=colors, s=100, alpha=0.9, zorder=5)
            
            ax.set_yticks(y_pos)
            self.set_colored_yticklabels(ax, sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{task.title()} Task - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (val, method) in enumerate(zip(sorted_data[metric], sorted_data['method_clean'])):
                ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_latent_comparison_plots(self, df: pd.DataFrame):
        """Create individual latent space metrics comparison plots."""
        eval_data = df[df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        # Key latent metrics to compare
        latent_metrics = ['hsic_global_score', 'active_units', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
        
        for metric in latent_metrics:
            data_to_plot = eval_data.dropna(subset=[metric])
            if not data_to_plot.empty:
                self._create_latent_metric_plot(data_to_plot, metric)
        
        # Enhanced variance analysis plots
        self._create_variance_analysis_plots(eval_data)
        
        # Per-dimension MI analysis plots
        self._create_mi_per_dimension_plots(eval_data)
        
        # Dimension efficiency analysis
        self._create_dimension_efficiency_plots(eval_data)
        
        # Variance vs MI correlation analysis
        self._create_variance_mi_correlation_plots(eval_data)
        
        # Comprehensive clustering quality analysis
        self._create_clustering_quality_analysis(eval_data)
        
        # Comprehensive geometric properties analysis
        self._create_geometric_properties_analysis(eval_data)
        
        # Enhanced information content analysis
        self._create_information_content_analysis(eval_data)
        
        print("  ✓ Saved enhanced latent comparison plots:")
        print("    - Comprehensive variance analysis (distribution, heatmaps, entropy)")
        print("    - Per-dimension MI heatmaps for each task")
        print("    - MI concentration and dimension ranking analysis")
        print("    - Multi-dimensional efficiency analysis")
        print("    - Variance-MI correlation analysis")
        print("    - Cross-task dimension specialization insights")
        print("    - Complete clustering quality analysis (all 3 metrics)")
        print("    - Comprehensive geometric properties analysis (all 3 metrics)")
        print("    - Enhanced information content analysis (HSIC + MI relationships)")
        
        # Create category summary comparison plots
        self._create_category_summary_plots(eval_data)
    
    def _create_latent_metric_plot(self, data: pd.DataFrame, metric: str):
        """Create individual latent metric plots with different chart types."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if metric == 'active_units':
            # Bubble plot for active units vs total dimensions
            bubble_data = data.dropna(subset=['active_units', 'dim'])
            if not bubble_data.empty:
                efficiency = bubble_data['active_units'] / bubble_data['dim']
                scatter = ax.scatter(bubble_data['dim'], bubble_data['active_units'], 
                                   s=efficiency*500, alpha=0.6, c='#5B9BD5')
                
                # Add method labels
                for i, (_, row) in enumerate(bubble_data.iterrows()):
                    ax.annotate(row['method_clean'], 
                               (row['dim'] + 5, row['active_units'] + 5),
                               fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Total Dimensions')
                ax.set_ylabel('Active Units')
                ax.set_title('Active Units vs Total Dimensions\n(Bubble size = Efficiency)', 
                            fontsize=16, fontweight='bold')
                
                # Add diagonal line for reference
                max_dim = bubble_data['dim'].max()
                ax.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Perfect Efficiency')
                ax.legend()
        
        elif metric in ['silhouette', 'trustworthiness', 'continuity', 'dist_corr']:
            # Dot plot for quality metrics
            sorted_data = data.sort_values(metric)
            y_pos = range(len(sorted_data))
            
            # Create dot plot
            colors = ['#5B9BD5'] * len(sorted_data)
            ax.scatter(sorted_data[metric], y_pos, color=colors, s=150, alpha=0.8)
            
            # Add connecting lines
            for i, val in enumerate(sorted_data[metric]):
                ax.plot([0, val], [i, i], color='gray', alpha=0.3, linewidth=1)
            
            ax.set_yticks(y_pos)
            self.set_colored_yticklabels(ax, sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Latent Space Quality - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (val, method) in enumerate(zip(sorted_data[metric], sorted_data['method_clean'])):
                ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontweight='bold')
        
        else:  # hsic_global_score
            # Horizontal bar plot for HSIC score
            sorted_data = data.sort_values(metric, ascending=True)
            colors = ['#5B9BD5'] * len(sorted_data)
            
            bars = ax.barh(range(len(sorted_data)), sorted_data[metric], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sorted_data)))
            self.set_colored_yticklabels(ax, sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Feature Independence - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_data[metric])):
                ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'latent_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_variance_analysis_plots(self, data: pd.DataFrame):
        """Create comprehensive variance analysis plots."""
        # Plot 1: Enhanced variance distribution with concentration metrics
        self._create_enhanced_variance_distribution_plot(data)
        
        # Plot 2: Per-dimension variance heatmap
        self._create_variance_heatmap(data)
        
        # Plot 3: Variance entropy and effective dimensionality
        self._create_variance_entropy_plot(data)
    
    def _create_enhanced_variance_distribution_plot(self, data: pd.DataFrame):
        """Create enhanced variance distribution plot with multiple metrics."""
        methods_with_variance = data.dropna(subset=['variance_mean', 'variance_std', 'variance_concentration'])
        
        if not methods_with_variance.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Variance Analysis', fontsize=18, fontweight='bold')
            
            methods = methods_with_variance['method_clean']
            x_pos = range(len(methods))
            
            # Plot 1: Mean vs Std
            colors1 = ['#5B9BD5'] * len(methods_with_variance)
            scatter1 = ax1.scatter(methods_with_variance['variance_mean'], 
                                 methods_with_variance['variance_std'],
                                 c=colors1, s=150, alpha=0.8, edgecolors='black')
            ax1.set_xlabel('Mean Variance per Dimension')
            ax1.set_ylabel('Std Variance per Dimension')
            ax1.set_title('Variance Mean vs Std\n(Color = Concentration)')
            
            # Add method labels
            for i, (mean_var, std_var, method) in enumerate(zip(methods_with_variance['variance_mean'],
                                                              methods_with_variance['variance_std'],
                                                              methods)):
                ax1.annotate(method, (mean_var, std_var), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # Plot 2: Variance Concentration
            colors = self.get_professional_colors()
            bars2 = ax2.bar(x_pos, methods_with_variance['variance_concentration'], 
                          alpha=0.8, color=colors['primary_purple'])
            ax2.set_xticks(x_pos)
            self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right')
            ax2.set_ylabel('Variance Concentration (Top 20%)')
            ax2.set_title('Variance Concentration in Top Dimensions')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars2, methods_with_variance['variance_concentration']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Coefficient of Variation
            cv_data = methods_with_variance.dropna(subset=['variance_cv'])
            if not cv_data.empty:
                bars3 = ax3.bar(range(len(cv_data)), cv_data['variance_cv'], 
                              alpha=0.8, color=colors['primary_blue'])
                ax3.set_xticks(range(len(cv_data)))
                self.set_colored_xticklabels(ax3, cv_data['method_clean'], rotation=45, ha='right')
                ax3.set_ylabel('Coefficient of Variation')
                ax3.set_title('Variance Distribution Uniformity')
                ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Min vs Max Variance
            minmax_data = methods_with_variance.dropna(subset=['variance_min', 'variance_max'])
            if not minmax_data.empty:
                width = 0.35
                x_pos4 = np.arange(len(minmax_data))
                bars4a = ax4.bar(x_pos4 - width/2, minmax_data['variance_min'], 
                               width, label='Min Variance', alpha=0.8, color=colors['primary_blue'])
                bars4b = ax4.bar(x_pos4 + width/2, minmax_data['variance_max'], 
                               width, label='Max Variance', alpha=0.8, color=colors['primary_purple'])
                ax4.set_xticks(x_pos4)
                self.set_colored_xticklabels(ax4, minmax_data['method_clean'], rotation=45, ha='right')
                ax4.set_ylabel('Variance Value')
                ax4.set_title('Min vs Max Dimension Variance')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_heatmap(self, data: pd.DataFrame):
        """Create heatmap showing per-dimension variance across methods."""
        # Collect variance per dimension data
        variance_matrix = []
        method_names = []
        max_dims = 0
        
        for _, row in data.iterrows():
            if 'variance_per_dim' in row and row['variance_per_dim'] and len(row['variance_per_dim']) > 0:
                variance_matrix.append(row['variance_per_dim'])
                method_names.append(row['method_clean'])
                max_dims = max(max_dims, len(row['variance_per_dim']))
        
        if variance_matrix and max_dims > 1:
            # Pad shorter variance vectors with NaN
            variance_matrix_padded = []
            for var_vec in variance_matrix:
                padded = var_vec + [np.nan] * (max_dims - len(var_vec))
                variance_matrix_padded.append(padded)
            
            variance_array = np.array(variance_matrix_padded)
            
            fig, ax = plt.subplots(figsize=(max(8, max_dims), max(6, len(method_names)*0.8)))
            
            # Create heatmap
            im = ax.imshow(variance_array, cmap='YlOrRd', aspect='auto', 
                          interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(max_dims))
            ax.set_xticklabels([f'Dim {i+1}' for i in range(max_dims)])
            ax.set_yticks(range(len(method_names)))
            self.set_colored_yticklabels(ax, method_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Variance', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(method_names)):
                for j in range(max_dims):
                    if not np.isnan(variance_array[i, j]):
                        text = ax.text(j, i, f'{variance_array[i, j]:.2f}',
                                     ha='center', va='center', fontsize=8,
                                     color='white' if variance_array[i, j] > np.nanmean(variance_array) else 'black')
            
            ax.set_title('Per-Dimension Variance Heatmap', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Latent Dimensions')
            ax.set_ylabel('Methods')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_per_dimension_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_entropy_plot(self, data: pd.DataFrame):
        """Create plot showing variance entropy and effective dimensionality."""
        entropy_data = data.dropna(subset=['variance_entropy', 'effective_dim_variance'])
        
        if not entropy_data.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            methods = entropy_data['method_clean']
            x_pos = range(len(methods))
            
            # Plot 1: Variance Entropy
            colors = self.get_professional_colors()
            bars1 = ax1.bar(x_pos, entropy_data['variance_entropy'], 
                           alpha=0.8, color=colors['primary_blue'])
            ax1.set_xticks(x_pos)
            self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right')
            ax1.set_ylabel('Variance Entropy')
            ax1.set_title('Variance Distribution Entropy\n(Higher = More Uniform)')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars1, entropy_data['variance_entropy']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Effective Dimensionality vs Actual Dimensionality
            actual_dims = entropy_data['dim']
            effective_dims = entropy_data['effective_dim_variance']
            
            # Scatter plot with efficiency line
            scatter = ax2.scatter(actual_dims, effective_dims, 
                                s=150, alpha=0.7, c='#5B9BD5', 
                                edgecolors='black')
            
            # Add diagonal line for reference (perfect efficiency)
            max_dim = max(actual_dims.max(), effective_dims.max())
            ax2.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, 
                   label='Perfect Efficiency')
            
            # Add method labels
            for i, (actual, effective, method) in enumerate(zip(actual_dims, effective_dims, methods)):
                ax2.annotate(method, (actual, effective), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Actual Dimensions')
            ax2.set_ylabel('Effective Dimensions (Variance-based)')
            ax2.set_title('Dimensional Efficiency\n(Closer to diagonal = More efficient)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_entropy_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_mi_per_dimension_plots(self, data: pd.DataFrame):
        """Create per-dimension mutual information analysis plots."""
        # Create MI heatmaps for each task
        for task in ['gender', 'abnormal', 'age']:
            self._create_mi_heatmap(data, task)
        
        # Create MI concentration analysis
        self._create_mi_concentration_plot(data)
        
        # Create dimension ranking by informativeness
        self._create_dimension_ranking_plot(data)
    
    def _create_mi_heatmap(self, data: pd.DataFrame, task: str):
        """Create heatmap showing per-dimension MI for a specific task."""
        mi_col = f'mi_{task}_per_dim'
        
        # Collect MI per dimension data
        mi_matrix = []
        method_names = []
        max_dims = 0
        
        for _, row in data.iterrows():
            if mi_col in row and row[mi_col] and len(row[mi_col]) > 0:
                mi_matrix.append(row[mi_col])
                method_names.append(row['method_clean'])
                max_dims = max(max_dims, len(row[mi_col]))
        
        if mi_matrix and max_dims > 1:
            # Pad shorter MI vectors with NaN
            mi_matrix_padded = []
            for mi_vec in mi_matrix:
                padded = mi_vec + [np.nan] * (max_dims - len(mi_vec))
                mi_matrix_padded.append(padded)
            
            mi_array = np.array(mi_matrix_padded)
            
            fig, ax = plt.subplots(figsize=(max(8, max_dims), max(6, len(method_names)*0.8)))
            
            # Create heatmap with diverging colormap
            im = ax.imshow(mi_array, cmap='Reds', aspect='auto', 
                          interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(max_dims))
            ax.set_xticklabels([f'Dim {i+1}' for i in range(max_dims)])
            ax.set_yticks(range(len(method_names)))
            self.set_colored_yticklabels(ax, method_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mutual Information', rotation=270, labelpad=20)
            
            # Add text annotations for non-NaN values
            for i in range(len(method_names)):
                for j in range(max_dims):
                    if not np.isnan(mi_array[i, j]):
                        text = ax.text(j, i, f'{mi_array[i, j]:.3f}',
                                     ha='center', va='center', fontsize=8,
                                     color='white' if mi_array[i, j] > np.nanmean(mi_array) else 'black')
            
            ax.set_title(f'Per-Dimension Mutual Information: {task.title()} Task', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Latent Dimensions')
            ax.set_ylabel('Methods')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'mi_per_dimension_{task}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_mi_concentration_plot(self, data: pd.DataFrame):
        """Create plot showing MI concentration across tasks."""
        tasks = ['gender', 'abnormal']
        concentration_data = []
        
        for _, row in data.iterrows():
            row_data = {'method': row['method_clean']}
            for task in tasks:
                conc_col = f'mi_{task}_concentration'
                if conc_col in row and not pd.isna(row[conc_col]):
                    row_data[f'{task}_concentration'] = row[conc_col]
            
            if len(row_data) > 1:  # Has at least one concentration value
                concentration_data.append(row_data)
        
        if concentration_data:
            conc_df = pd.DataFrame(concentration_data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(conc_df))
            width = 0.35
            
            # Plot concentrations for each task
            colors = ['#5B9BD5', '#87CEEB']
            bars = []
            for i, task in enumerate(tasks):
                conc_col = f'{task}_concentration'
                if conc_col in conc_df.columns:
                    task_data = conc_df.dropna(subset=[conc_col])
                    if not task_data.empty:
                        task_bars = ax.bar(x_pos[task_data.index] + i*width, 
                                         task_data[conc_col], width, 
                                         label=f'{task.title()} Task', 
                                         alpha=0.8, color=colors[i])
                        bars.extend(task_bars)
            
            ax.set_xticks(x_pos + width/2)
            self.set_colored_xticklabels(ax, conc_df['method'], rotation=45, ha='right')
            ax.set_ylabel('MI Concentration (Top 20% Dimensions)')
            ax.set_title('Mutual Information Concentration Analysis', 
                        fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mi_concentration_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_dimension_ranking_plot(self, data: pd.DataFrame):
        """Create dimension ranking plot by informativeness."""
        # Find method with highest average MI for detailed analysis
        best_method_data = None
        best_avg_mi = 0
        
        for _, row in data.iterrows():
            avg_mi = 0
            count = 0
            for task in ['gender', 'abnormal']:
                mi_mean_col = f'mi_{task}_mean'
                if mi_mean_col in row and not pd.isna(row[mi_mean_col]):
                    avg_mi += row[mi_mean_col]
                    count += 1
            
            if count > 0:
                avg_mi /= count
                if avg_mi > best_avg_mi:
                    best_avg_mi = avg_mi
                    best_method_data = row
        
        # Skip single-method dimension ranking - focus on comparative analysis across methods
        # if best_method_data is not None:
            
        # High information dimensions across all methods comparison
        high_info_counts = []
        method_names = []
        
        for _, row in data.iterrows():
            total_high_info = 0
            for task in ['gender', 'abnormal']:
                high_info_col = f'mi_{task}_high_info_dims'
                if high_info_col in row and not pd.isna(row[high_info_col]):
                    total_high_info += row[high_info_col]
            
            if total_high_info > 0:
                high_info_counts.append(total_high_info)
                method_names.append(row['method_clean'])
        
        if high_info_counts:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Sort by count for better visualization
            sorted_data = sorted(zip(method_names, high_info_counts), key=lambda x: x[1], reverse=True)
            method_names_sorted, high_info_counts_sorted = zip(*sorted_data)
            
            colors = self.get_professional_colors()
            bars = ax.bar(range(len(method_names_sorted)), high_info_counts_sorted, 
                         alpha=0.8, color=colors['primary_purple'])
            ax.set_xticks(range(len(method_names_sorted)))
            self.set_colored_xticklabels(ax, method_names_sorted, rotation=45, ha='right')
            ax.set_ylabel('High Information Dimensions')
            ax.set_title('Count of Highly Informative Dimensions Across Methods')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, high_info_counts_sorted):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(val)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dimension_ranking_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_dimension_efficiency_plots(self, data: pd.DataFrame):
        """Create comprehensive dimension efficiency analysis plots."""
        # Plot 1: Efficiency vs Performance
        self._create_efficiency_performance_plot(data)
        
        # Plot 2: Multi-dimensional efficiency analysis
        self._create_multidim_efficiency_plot(data)
    
    def _create_efficiency_performance_plot(self, data: pd.DataFrame):
        """Create scatter plot showing efficiency vs classification performance."""
        # Collect efficiency and performance data
        efficiency_data = []
        
        for _, row in data.iterrows():
            if ('active_units' in row and 'dim' in row and 
                not pd.isna(row['active_units']) and not pd.isna(row['dim']) and
                row['dim'] > 0):
                
                row_data = {
                    'method': row['method_clean'],
                    'efficiency': row['active_units'] / row['dim'],
                    'active_units': row['active_units'],
                    'total_dims': row['dim']
                }
                
                # Add MI data
                for task in ['gender', 'abnormal']:
                    mi_col = f'mi_{task}_mean'
                    if mi_col in row and not pd.isna(row[mi_col]):
                        row_data[f'mi_{task}'] = row[mi_col]
                
                efficiency_data.append(row_data)
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Dimension Efficiency Analysis', fontsize=18, fontweight='bold')
            
            # Plot 1: Efficiency vs Gender MI
            if 'mi_gender' in eff_df.columns:
                gender_data = eff_df.dropna(subset=['efficiency', 'mi_gender'])
                if not gender_data.empty:
                    scatter1 = ax1.scatter(gender_data['efficiency'], gender_data['mi_gender'],
                                         s=gender_data['active_units']*10, alpha=0.7,
                                         c='#5B9BD5',
                                         edgecolors='black')
                    
                    for i, (eff, mi, method) in enumerate(zip(gender_data['efficiency'], 
                                                            gender_data['mi_gender'],
                                                            gender_data['method'])):
                        ax1.annotate(method, (eff, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)
                    
                    ax1.set_xlabel('Dimensional Efficiency')
                    ax1.set_ylabel('Gender MI')
                    ax1.set_title('Efficiency vs Gender Informativeness\n(Bubble size = Active Units)')
                    ax1.grid(True, alpha=0.3)
            
            # Plot 2: Efficiency vs Abnormal MI
            if 'mi_abnormal' in eff_df.columns:
                abnormal_data = eff_df.dropna(subset=['efficiency', 'mi_abnormal'])
                if not abnormal_data.empty:
                    scatter2 = ax2.scatter(abnormal_data['efficiency'], abnormal_data['mi_abnormal'],
                                         s=abnormal_data['active_units']*10, alpha=0.7,
                                         c='#5B9BD5',
                                         edgecolors='black')
                    
                    for i, (eff, mi, method) in enumerate(zip(abnormal_data['efficiency'], 
                                                            abnormal_data['mi_abnormal'],
                                                            abnormal_data['method'])):
                        ax2.annotate(method, (eff, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)
                    
                    ax2.set_xlabel('Dimensional Efficiency')
                    ax2.set_ylabel('Abnormal MI')
                    ax2.set_title('Efficiency vs Abnormal Informativeness\n(Bubble size = Active Units)')
                    ax2.grid(True, alpha=0.3)
            
            # Plot 3: Active units vs Total dimensions
            scatter3 = ax3.scatter(eff_df['total_dims'], eff_df['active_units'],
                                 s=150, alpha=0.7, c='#5B9BD5',
                                 edgecolors='black')
            
            # Add diagonal line
            max_dim = max(eff_df['total_dims'].max(), eff_df['active_units'].max())
            ax3.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Perfect Efficiency')
            
            for i, (total, active, method) in enumerate(zip(eff_df['total_dims'], 
                                                          eff_df['active_units'],
                                                          eff_df['method'])):
                ax3.annotate(method, (total, active), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
            
            ax3.set_xlabel('Total Dimensions')
            ax3.set_ylabel('Active Units')
            ax3.set_title('Active vs Total Dimensions\n(Color = Efficiency)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar for efficiency
            cbar3 = plt.colorbar(scatter3, ax=ax3)
            cbar3.set_label('Efficiency', rotation=270, labelpad=20)
            
            # Plot 4: Efficiency distribution
            bars4 = ax4.bar(range(len(eff_df)), eff_df['efficiency'], 
                          alpha=0.8, color='#5B9BD5')
            ax4.set_xticks(range(len(eff_df)))
            self.set_colored_xticklabels(ax4, eff_df['method'], rotation=45, ha='right')
            ax4.set_ylabel('Dimensional Efficiency')
            ax4.set_title('Efficiency Comparison Across Methods')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars4, eff_df['efficiency']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dimension_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_mi_correlation_plots(self, data: pd.DataFrame):
        """Create plots showing correlation between variance and mutual information."""
        # Analyze correlation between variance and MI per dimension
        correlation_data = []
        
        for _, row in data.iterrows():
            if ('variance_per_dim' in row and row['variance_per_dim'] and
                len(row['variance_per_dim']) > 0):
                
                variance_vec = np.array(row['variance_per_dim'])
                method_name = row['method_clean']
                
                for task in ['gender', 'abnormal']:
                    mi_col = f'mi_{task}_per_dim'
                    if (mi_col in row and row[mi_col] and 
                        len(row[mi_col]) == len(variance_vec)):
                        
                        mi_vec = np.array(row[mi_col])
                        
                        # Calculate correlation
                        if len(variance_vec) > 1 and np.std(variance_vec) > 0 and np.std(mi_vec) > 0:
                            correlation = np.corrcoef(variance_vec, mi_vec)[0, 1]
                            
                            correlation_data.append({
                                'method': method_name,
                                'task': task,
                                'correlation': correlation,
                                'variance_vec': variance_vec,
                                'mi_vec': mi_vec
                            })
        
        if correlation_data:
            # Plot 1: Correlation heatmap
            corr_df = pd.DataFrame(correlation_data)
            
            # Create pivot table for heatmap
            pivot_corr = corr_df.pivot(index='method', columns='task', values='correlation')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Correlation heatmap
            if not pivot_corr.empty:
                sns.heatmap(pivot_corr, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.2f', cbar_kws={"shrink": .8},
                           linewidths=0.5, ax=ax1)
                ax1.set_title('Variance-MI Correlation by Method and Task', 
                            fontsize=14, fontweight='bold')
                ax1.set_xlabel('Task')
                ax1.set_ylabel('Method')
            
            # Scatter plot example (best correlated method)
            if 'correlation' in corr_df.columns:
                best_corr_idx = corr_df['correlation'].abs().idxmax()
                best_data = corr_df.loc[best_corr_idx]
                
                ax2.scatter(best_data['variance_vec'], best_data['mi_vec'],
                           alpha=0.7, s=100, color='#5B9BD5')
                
                # Add trend line
                z = np.polyfit(best_data['variance_vec'], best_data['mi_vec'], 1)
                p = np.poly1d(z)
                ax2.plot(best_data['variance_vec'], p(best_data['variance_vec']), 
                        "r--", alpha=0.8, linewidth=2)
                
                ax2.set_xlabel('Variance per Dimension')
                ax2.set_ylabel('MI per Dimension')
                ax2.set_title(f'Variance vs MI Example\n{best_data["method"]} - {best_data["task"].title()} Task\n(r = {best_data["correlation"]:.3f})')
                ax2.grid(True, alpha=0.3)
                
                # Annotate dimensions
                for i, (var, mi) in enumerate(zip(best_data['variance_vec'], best_data['mi_vec'])):
                    ax2.annotate(f'D{i+1}', (var, mi), xytext=(3, 3),
                               textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_mi_correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_multidim_efficiency_plot(self, data: pd.DataFrame):
        """Create multi-dimensional efficiency analysis plot."""
        # Comprehensive efficiency metrics
        efficiency_metrics = []
        
        for _, row in data.iterrows():
            if ('active_units' in row and 'dim' in row and 'variance_entropy' in row and
                not pd.isna(row['active_units']) and not pd.isna(row['dim']) and 
                not pd.isna(row['variance_entropy']) and row['dim'] > 0):
                
                metrics = {
                    'method': row['method_clean'],
                    'dimensional_efficiency': row['active_units'] / row['dim'],
                    'variance_entropy': row['variance_entropy'],
                    'active_units': row['active_units'],
                    'total_dims': row['dim']
                }
                
                # Add variance concentration if available
                if 'variance_concentration' in row and not pd.isna(row['variance_concentration']):
                    metrics['variance_concentration'] = row['variance_concentration']
                
                # Add MI concentration if available
                mi_concentration_sum = 0
                mi_count = 0
                for task in ['gender', 'abnormal']:
                    conc_col = f'mi_{task}_concentration'
                    if conc_col in row and not pd.isna(row[conc_col]):
                        mi_concentration_sum += row[conc_col]
                        mi_count += 1
                
                if mi_count > 0:
                    metrics['mi_concentration'] = mi_concentration_sum / mi_count
                
                efficiency_metrics.append(metrics)
        
        if efficiency_metrics:
            eff_df = pd.DataFrame(efficiency_metrics)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create bubble plot with multiple dimensions
            x = eff_df['dimensional_efficiency']
            y = eff_df['variance_entropy']
            
            # Size based on total dimensions
            sizes = eff_df['total_dims'] * 20
            
            # Color based on MI concentration if available
            if 'mi_concentration' in eff_df.columns:
                color_values = '#5B9BD5'
                colormap = None
                color_label = 'Methods'
            else:
                color_values = '#5B9BD5'
                colormap = None
                color_label = 'Methods'
            
            scatter = ax.scatter(x, y, s=sizes, c=color_values, alpha=0.7, 
                               cmap=colormap, edgecolors='black', linewidth=1)
            
            # Add method labels
            for i, (eff, entropy, method) in enumerate(zip(x, y, eff_df['method'])):
                ax.annotate(method, (eff, entropy), xytext=(5, 5),
                           textcoords='offset points', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Dimensional Efficiency (Active/Total)', fontsize=12)
            ax.set_ylabel('Variance Entropy (Uniformity)', fontsize=12)
            ax.set_title('Multi-Dimensional Efficiency Analysis\n(Bubble size = Total Dimensions)', 
                        fontsize=16, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_label, rotation=270, labelpad=20)
            
            # Add grid and reference lines
            ax.grid(True, alpha=0.3)
            
            # Add quadrant labels for interpretation
            ax.axhline(y=np.median(y), color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=np.median(x), color='red', linestyle='--', alpha=0.5)
            
            # Add quadrant annotations
            colors = self.get_professional_colors()
            ax.text(0.95, 0.95, 'High Efficiency\nHigh Uniformity', transform=ax.transAxes,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor=colors['primary_blue'], alpha=0.7))
            ax.text(0.05, 0.05, 'Low Efficiency\nLow Uniformity', transform=ax.transAxes,
                   ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor=colors['primary_purple'], alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'multidimensional_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_clustering_quality_analysis(self, data: pd.DataFrame):
        """Create comprehensive clustering quality analysis with all three metrics."""
        # Collect clustering quality data
        clustering_data = []
        
        for _, row in data.iterrows():
            if ('silhouette' in row and 'davies_bouldin' in row and 'calinski_harabasz' in row and
                not pd.isna(row['silhouette']) and not pd.isna(row['davies_bouldin']) and 
                not pd.isna(row['calinski_harabasz'])):
                
                clustering_data.append({
                    'method': row['method_clean'],
                    'silhouette': row['silhouette'],
                    'davies_bouldin': row['davies_bouldin'],
                    'calinski_harabasz': row['calinski_harabasz']
                })
        
        if clustering_data:
            cluster_df = pd.DataFrame(clustering_data)
            
            # Create comprehensive clustering analysis plot
            fig = plt.figure(figsize=(24, 16))
            
            # Create subplot layout: 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)
            
            # Plot 1: Silhouette Score (Higher Better)
            ax1 = fig.add_subplot(gs[0, 0])
            sorted_silhouette = cluster_df.sort_values('silhouette', ascending=True)
            colors1 = ['#5B9BD5'] * len(sorted_silhouette)
            bars1 = ax1.barh(range(len(sorted_silhouette)), sorted_silhouette['silhouette'], 
                           color=colors1, alpha=0.8)
            ax1.set_yticks(range(len(sorted_silhouette)))
            self.set_colored_yticklabels(ax1, sorted_silhouette['method'])
            ax1.set_xlabel('Silhouette Score')
            ax1.set_title('Clustering Quality: Silhouette Score\n(Higher = Better Separated Clusters)', 
                         fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, sorted_silhouette['silhouette'])):
                ax1.text(val + max(sorted_silhouette['silhouette'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Davies-Bouldin Index (Lower Better)
            ax2 = fig.add_subplot(gs[0, 1])
            sorted_davies = cluster_df.sort_values('davies_bouldin', ascending=False)  # Sort desc for visual consistency
            colors2 = ['#5B9BD5'] * len(sorted_davies)
            bars2 = ax2.barh(range(len(sorted_davies)), sorted_davies['davies_bouldin'], 
                           color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(sorted_davies)))
            self.set_colored_yticklabels(ax2, sorted_davies['method'])
            ax2.set_xlabel('Davies-Bouldin Index')
            ax2.set_title('Clustering Quality: Davies-Bouldin Index\n(Lower = Better Separated Clusters)', 
                         fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars2, sorted_davies['davies_bouldin'])):
                ax2.text(val + max(sorted_davies['davies_bouldin'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Calinski-Harabasz Index (Higher Better)
            ax3 = fig.add_subplot(gs[0, 2])
            sorted_calinski = cluster_df.sort_values('calinski_harabasz', ascending=True)
            colors3 = ['#5B9BD5'] * len(sorted_calinski)
            bars3 = ax3.barh(range(len(sorted_calinski)), sorted_calinski['calinski_harabasz'], 
                           color=colors3, alpha=0.8)
            ax3.set_yticks(range(len(sorted_calinski)))
            self.set_colored_yticklabels(ax3, sorted_calinski['method'])
            ax3.set_xlabel('Calinski-Harabasz Index')
            ax3.set_title('Clustering Quality: Calinski-Harabasz Index\n(Higher = Better Separated Clusters)', 
                         fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars3, sorted_calinski['calinski_harabasz'])):
                ax3.text(val + max(sorted_calinski['calinski_harabasz'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Clustering Quality Radar Chart
            ax4 = fig.add_subplot(gs[1, :2], projection='polar')
            
            # Normalize metrics for radar chart (0-1 scale)
            normalized_data = cluster_df.copy()
            normalized_data['silhouette_norm'] = normalized_data['silhouette'] / normalized_data['silhouette'].max()
            normalized_data['davies_bouldin_norm'] = 1 - (normalized_data['davies_bouldin'] / normalized_data['davies_bouldin'].max())  # Invert since lower is better
            normalized_data['calinski_harabasz_norm'] = normalized_data['calinski_harabasz'] / normalized_data['calinski_harabasz'].max()
            
            # Set up radar chart
            metrics_radar = ['Silhouette\n(norm)', 'Davies-Bouldin\n(inverted)', 'Calinski-Harabasz\n(norm)']
            N = len(metrics_radar)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each method
            colors_radar = plt.cm.Set3(np.linspace(0, 1, len(normalized_data)))
            for idx, (_, row) in enumerate(normalized_data.iterrows()):
                values = [row['silhouette_norm'], row['davies_bouldin_norm'], row['calinski_harabasz_norm']]
                values += values[:1]  # Complete the circle
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=row['method'], 
                        color=colors_radar[idx], alpha=0.7)
                ax4.fill(angles, values, alpha=0.1, color=colors_radar[idx])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_radar)
            ax4.set_ylim(0, 1)
            ax4.set_title('Clustering Quality Radar Chart\n(All metrics normalized 0-1, higher = better)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Overall Clustering Quality Score
            ax5 = fig.add_subplot(gs[1, 2])
            
            # Calculate composite clustering score (average of normalized metrics)
            cluster_df['composite_score'] = (
                normalized_data['silhouette_norm'] + 
                normalized_data['davies_bouldin_norm'] + 
                normalized_data['calinski_harabasz_norm']
            ) / 3
            
            sorted_composite = cluster_df.sort_values('composite_score', ascending=True)
            colors5 = plt.cm.viridis(sorted_composite['composite_score'])
            bars5 = ax5.barh(range(len(sorted_composite)), sorted_composite['composite_score'], 
                           color=colors5, alpha=0.8)
            ax5.set_yticks(range(len(sorted_composite)))
            self.set_colored_yticklabels(ax5, sorted_composite['method'])
            ax5.set_xlabel('Composite Clustering Score')
            ax5.set_title('Overall Clustering Quality\n(Average of Normalized Metrics)', 
                         fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars5, sorted_composite['composite_score'])):
                ax5.text(val + max(sorted_composite['composite_score'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.suptitle('Comprehensive Clustering Quality Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'clustering_quality_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create clustering correlation plot
            self._create_clustering_correlation_plot(cluster_df)
    
    def _create_clustering_correlation_plot(self, cluster_df: pd.DataFrame):
        """Create correlation analysis between different clustering metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Metrics Correlation Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Silhouette vs Davies-Bouldin (should be negatively correlated)
        scatter1 = ax1.scatter(cluster_df['silhouette'], cluster_df['davies_bouldin'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='viridis',
                             edgecolors='black')
        
        # Add trend line
        z1 = np.polyfit(cluster_df['silhouette'], cluster_df['davies_bouldin'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(cluster_df['silhouette'], p1(cluster_df['silhouette']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr1 = np.corrcoef(cluster_df['silhouette'], cluster_df['davies_bouldin'])[0, 1]
        ax1.set_xlabel('Silhouette Score (Higher Better)')
        ax1.set_ylabel('Davies-Bouldin Index (Lower Better)')
        ax1.set_title(f'Silhouette vs Davies-Bouldin\nr = {corr1:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (sil, db, method) in enumerate(zip(cluster_df['silhouette'], 
                                                cluster_df['davies_bouldin'], 
                                                cluster_df['method'])):
            ax1.annotate(method, (sil, db), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 2: Silhouette vs Calinski-Harabasz (should be positively correlated)
        scatter2 = ax2.scatter(cluster_df['silhouette'], cluster_df['calinski_harabasz'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='plasma',
                             edgecolors='black')
        
        # Add trend line
        z2 = np.polyfit(cluster_df['silhouette'], cluster_df['calinski_harabasz'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(cluster_df['silhouette'], p2(cluster_df['silhouette']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr2 = np.corrcoef(cluster_df['silhouette'], cluster_df['calinski_harabasz'])[0, 1]
        ax2.set_xlabel('Silhouette Score (Higher Better)')
        ax2.set_ylabel('Calinski-Harabasz Index (Higher Better)')
        ax2.set_title(f'Silhouette vs Calinski-Harabasz\nr = {corr2:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (sil, ch, method) in enumerate(zip(cluster_df['silhouette'], 
                                                cluster_df['calinski_harabasz'], 
                                                cluster_df['method'])):
            ax2.annotate(method, (sil, ch), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 3: Davies-Bouldin vs Calinski-Harabasz (should be negatively correlated)
        scatter3 = ax3.scatter(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='coolwarm',
                             edgecolors='black')
        
        # Add trend line
        z3 = np.polyfit(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(cluster_df['davies_bouldin'], p3(cluster_df['davies_bouldin']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr3 = np.corrcoef(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'])[0, 1]
        ax3.set_xlabel('Davies-Bouldin Index (Lower Better)')
        ax3.set_ylabel('Calinski-Harabasz Index (Higher Better)')
        ax3.set_title(f'Davies-Bouldin vs Calinski-Harabasz\nr = {corr3:.3f}')
        ax3.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (db, ch, method) in enumerate(zip(cluster_df['davies_bouldin'], 
                                                cluster_df['calinski_harabasz'], 
                                                cluster_df['method'])):
            ax3.annotate(method, (db, ch), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 4: Correlation matrix heatmap
        corr_matrix = cluster_df[['silhouette', 'davies_bouldin', 'calinski_harabasz']].corr()
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        metrics_labels = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
        ax4.set_xticks(range(len(metrics_labels)))
        ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_labels)))
        ax4.set_yticklabels(metrics_labels)
        
        # Add correlation values
        for i in range(len(metrics_labels)):
            for j in range(len(metrics_labels)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax4.set_title('Clustering Metrics\nCorrelation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_metrics_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_geometric_properties_analysis(self, data: pd.DataFrame):
        """Create comprehensive geometric properties analysis with all three metrics."""
        # Collect geometric properties data
        geometric_data = []
        
        for _, row in data.iterrows():
            if ('trustworthiness' in row and 'continuity' in row and 'dist_corr' in row and
                not pd.isna(row['trustworthiness']) and not pd.isna(row['continuity']) and 
                not pd.isna(row['dist_corr'])):
                
                geometric_data.append({
                    'method': row['method_clean'],
                    'trustworthiness': row['trustworthiness'],
                    'continuity': row['continuity'],
                    'dist_corr': row['dist_corr']
                })
        
        if geometric_data:
            geom_df = pd.DataFrame(geometric_data)
            
            # Create comprehensive geometric analysis plot
            fig = plt.figure(figsize=(24, 16))
            
            # Create subplot layout: 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)
            
            # Plot 1: Trustworthiness (Higher Better)
            ax1 = fig.add_subplot(gs[0, 0])
            sorted_trust = geom_df.sort_values('trustworthiness', ascending=True)
            colors1 = plt.cm.RdYlGn(sorted_trust['trustworthiness'])
            bars1 = ax1.barh(range(len(sorted_trust)), sorted_trust['trustworthiness'], 
                           color=colors1, alpha=0.8)
            ax1.set_yticks(range(len(sorted_trust)))
            self.set_colored_yticklabels(ax1, sorted_trust['method'])
            ax1.set_xlabel('Trustworthiness Score')
            ax1.set_title('Geometric Quality: Trustworthiness\n(Higher = Better Neighborhood Preservation)', 
                         fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, sorted_trust['trustworthiness'])):
                ax1.text(val + max(sorted_trust['trustworthiness'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Continuity (Higher Better)
            ax2 = fig.add_subplot(gs[0, 1])
            sorted_cont = geom_df.sort_values('continuity', ascending=True)
            colors2 = plt.cm.RdYlGn(sorted_cont['continuity'])
            bars2 = ax2.barh(range(len(sorted_cont)), sorted_cont['continuity'], 
                           color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(sorted_cont)))
            self.set_colored_yticklabels(ax2, sorted_cont['method'])
            ax2.set_xlabel('Continuity Score')
            ax2.set_title('Geometric Quality: Continuity\n(Higher = Better Smoothness)', 
                         fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars2, sorted_cont['continuity'])):
                ax2.text(val + max(sorted_cont['continuity'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Distance Correlation (Higher Better)
            ax3 = fig.add_subplot(gs[0, 2])
            sorted_dist = geom_df.sort_values('dist_corr', ascending=True)
            colors3 = plt.cm.RdYlGn(sorted_dist['dist_corr'])
            bars3 = ax3.barh(range(len(sorted_dist)), sorted_dist['dist_corr'], 
                           color=colors3, alpha=0.8)
            ax3.set_yticks(range(len(sorted_dist)))
            self.set_colored_yticklabels(ax3, sorted_dist['method'])
            ax3.set_xlabel('Distance Correlation')
            ax3.set_title('Geometric Quality: Distance Correlation\n(Higher = Better Distance Preservation)', 
                         fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars3, sorted_dist['dist_corr'])):
                ax3.text(val + max(sorted_dist['dist_corr'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Geometric Quality Radar Chart
            ax4 = fig.add_subplot(gs[1, :2], projection='polar')
            
            # Set up radar chart
            metrics_radar = ['Trustworthiness', 'Continuity', 'Distance Correlation']
            N = len(metrics_radar)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each method
            colors_radar = plt.cm.Set3(np.linspace(0, 1, len(geom_df)))
            for idx, (_, row) in enumerate(geom_df.iterrows()):
                values = [row['trustworthiness'], row['continuity'], row['dist_corr']]
                values += values[:1]  # Complete the circle
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=row['method'], 
                        color=colors_radar[idx], alpha=0.7)
                ax4.fill(angles, values, alpha=0.1, color=colors_radar[idx])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_radar)
            ax4.set_ylim(0, 1)
            ax4.set_title('Geometric Properties Radar Chart\n(All metrics 0-1, higher = better)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Overall Geometric Quality Score
            ax5 = fig.add_subplot(gs[1, 2])
            
            # Calculate composite geometric score (average of all metrics)
            geom_df['composite_geometric_score'] = (
                geom_df['trustworthiness'] + 
                geom_df['continuity'] + 
                geom_df['dist_corr']
            ) / 3
            
            sorted_composite = geom_df.sort_values('composite_geometric_score', ascending=True)
            colors5 = plt.cm.viridis(sorted_composite['composite_geometric_score'])
            bars5 = ax5.barh(range(len(sorted_composite)), sorted_composite['composite_geometric_score'], 
                           color=colors5, alpha=0.8)
            ax5.set_yticks(range(len(sorted_composite)))
            self.set_colored_yticklabels(ax5, sorted_composite['method'])
            ax5.set_xlabel('Composite Geometric Score')
            ax5.set_title('Overall Geometric Quality\n(Average of All Metrics)', 
                         fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars5, sorted_composite['composite_geometric_score'])):
                ax5.text(val + max(sorted_composite['composite_geometric_score'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.suptitle('Comprehensive Geometric Properties Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'geometric_properties_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create geometric correlation plot
            self._create_geometric_correlation_plot(geom_df)
    
    def _create_geometric_correlation_plot(self, geom_df: pd.DataFrame):
        """Create correlation analysis between different geometric metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geometric Properties Correlation Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Trustworthiness vs Continuity (should be positively correlated)
        scatter1 = ax1.scatter(geom_df['trustworthiness'], geom_df['continuity'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='viridis',
                             edgecolors='black')
        
        # Add trend line
        z1 = np.polyfit(geom_df['trustworthiness'], geom_df['continuity'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(geom_df['trustworthiness'], p1(geom_df['trustworthiness']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr1 = np.corrcoef(geom_df['trustworthiness'], geom_df['continuity'])[0, 1]
        ax1.set_xlabel('Trustworthiness (Neighborhood Preservation)')
        ax1.set_ylabel('Continuity (Smoothness)')
        ax1.set_title(f'Trustworthiness vs Continuity\nr = {corr1:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (trust, cont, method) in enumerate(zip(geom_df['trustworthiness'], 
                                                    geom_df['continuity'], 
                                                    geom_df['method'])):
            ax1.annotate(method, (trust, cont), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 2: Trustworthiness vs Distance Correlation
        scatter2 = ax2.scatter(geom_df['trustworthiness'], geom_df['dist_corr'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='plasma',
                             edgecolors='black')
        
        # Add trend line
        z2 = np.polyfit(geom_df['trustworthiness'], geom_df['dist_corr'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(geom_df['trustworthiness'], p2(geom_df['trustworthiness']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr2 = np.corrcoef(geom_df['trustworthiness'], geom_df['dist_corr'])[0, 1]
        ax2.set_xlabel('Trustworthiness (Neighborhood Preservation)')
        ax2.set_ylabel('Distance Correlation')
        ax2.set_title(f'Trustworthiness vs Distance Correlation\nr = {corr2:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (trust, dist, method) in enumerate(zip(geom_df['trustworthiness'], 
                                                    geom_df['dist_corr'], 
                                                    geom_df['method'])):
            ax2.annotate(method, (trust, dist), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 3: Continuity vs Distance Correlation
        scatter3 = ax3.scatter(geom_df['continuity'], geom_df['dist_corr'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='coolwarm',
                             edgecolors='black')
        
        # Add trend line
        z3 = np.polyfit(geom_df['continuity'], geom_df['dist_corr'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(geom_df['continuity'], p3(geom_df['continuity']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr3 = np.corrcoef(geom_df['continuity'], geom_df['dist_corr'])[0, 1]
        ax3.set_xlabel('Continuity (Smoothness)')
        ax3.set_ylabel('Distance Correlation')
        ax3.set_title(f'Continuity vs Distance Correlation\nr = {corr3:.3f}')
        ax3.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (cont, dist, method) in enumerate(zip(geom_df['continuity'], 
                                                   geom_df['dist_corr'], 
                                                   geom_df['method'])):
            ax3.annotate(method, (cont, dist), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 4: Correlation matrix heatmap
        corr_matrix = geom_df[['trustworthiness', 'continuity', 'dist_corr']].corr()
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        metrics_labels = ['Trustworthiness', 'Continuity', 'Distance Corr']
        ax4.set_xticks(range(len(metrics_labels)))
        ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_labels)))
        ax4.set_yticklabels(metrics_labels)
        
        # Add correlation values
        for i in range(len(metrics_labels)):
            for j in range(len(metrics_labels)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax4.set_title('Geometric Metrics\\nCorrelation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'geometric_properties_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========== SCIENTIFIC STYLING UTILITIES ==========
    
    def _apply_scientific_styling(self, ax, title_prefix=None):
        """Apply consistent scientific styling to a subplot."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
        
        # Make axis labels bold
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        
        # Add title prefix if provided
        if title_prefix and ax.get_title():
            current_title = ax.get_title()
            if not current_title.startswith(title_prefix):
                ax.set_title(f'{title_prefix}) {current_title}', fontweight='bold', pad=15)
        
        # Make x-tick labels bold
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
    
    # ========== CATEGORY SUMMARY COMPARISON PLOTS ==========
    
    def _create_category_summary_plots(self, data: pd.DataFrame):
        """Create summary comparison plots for each evaluation category with methods on X-axis."""
        print("\n  Creating category summary comparison plots...")
        
        # 1. Dimensionality & Efficiency Summary
        self._create_efficiency_summary_comparison(data)
        
        # 2. Information Content Summary
        self._create_information_summary_comparison(data)
        
        # 3. Feature Independence Summary
        self._create_independence_summary_comparison(data)
        # Also create single-metric, publication-style independence plots
        self._create_feature_dependence_plot(data)
        self._create_feature_structure_quality_plot(data)
        
        # 4. Geometric Preservation Summary
        self._create_geometric_summary_comparison(data)
        
        # 5. Cluster Quality Summary
        self._create_clustering_summary_comparison(data)
        
        print("  ✓ Saved category summary comparison plots")
        print("    - Classification Performance Summary")
        print("    - Dimensionality & Efficiency Summary")
        print("    - Information Content Summary")
        print("    - Feature Independence Summary")
        print("    - Geometric Preservation Summary")
        print("    - Cluster Quality Summary")
    
    def _create_efficiency_summary_comparison(self, data: pd.DataFrame):
        """Create dimensionality & efficiency summary comparison plot."""
        efficiency_data = data.dropna(subset=['dim', 'active_units']).copy()
        if efficiency_data.empty:
            return
            
        # Calculate efficiency metrics
        efficiency_data['dimensional_efficiency'] = efficiency_data['active_units'] / efficiency_data['dim']
        # Sort by canonical method order instead of alphabetical
        ordered_methods = self.get_canonical_method_order(efficiency_data['method_clean'].tolist())
        efficiency_data = efficiency_data.set_index('method_clean').reindex(ordered_methods).reset_index()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dimensionality & Efficiency: Method Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        methods = efficiency_data['method_clean']
        x_pos = range(len(methods))
        
        # Plot 1: Total Dimensions
        bars1 = ax1.bar(x_pos, efficiency_data['dim'], alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.8)
        ax1.set_xticks(x_pos)
        self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right', fontweight='bold')
        ax1.set_ylabel('Total Dimensions', fontweight='bold')
        ax1.set_title('A) Total Representational Capacity', fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        for i, (bar, val) in enumerate(zip(bars1, efficiency_data['dim'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Active Units
        bars2 = ax2.bar(x_pos, efficiency_data['active_units'], alpha=0.8, color='forestgreen', edgecolor='black', linewidth=0.8)
        ax2.set_xticks(x_pos)
        self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right', fontweight='bold')
        ax2.set_ylabel('Active Units', fontweight='bold')
        ax2.set_title('B) Utilized Dimensions', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        for i, (bar, val) in enumerate(zip(bars2, efficiency_data['active_units'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Dimensional Efficiency
        colors = self.get_professional_colors()
        bars3 = ax3.bar(x_pos, efficiency_data['dimensional_efficiency'], alpha=0.8, color=colors['primary_blue'], edgecolor='black', linewidth=0.8)
        ax3.set_xticks(x_pos)
        self.set_colored_xticklabels(ax3, methods, rotation=45, ha='right', fontweight='bold')
        ax3.set_ylabel('Efficiency Ratio', fontweight='bold')
        ax3.set_title('C) Dimensional Efficiency (Active/Total)', fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax3.set_ylim(0, 1.05)
        for i, (bar, val) in enumerate(zip(bars3, efficiency_data['dimensional_efficiency'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Variance Entropy (if available)
        if 'variance_entropy' in efficiency_data.columns:
            entropy_data = efficiency_data.dropna(subset=['variance_entropy'])
            if not entropy_data.empty:
                bars4 = ax4.bar(range(len(entropy_data)), entropy_data['variance_entropy'], 
                              alpha=0.8, color='coral')
                ax4.set_xticks(range(len(entropy_data)))
                self.set_colored_xticklabels(ax4, entropy_data['method_clean'], rotation=45, ha='right')
                ax4.set_ylabel('Variance Entropy')
                ax4.set_title('D) Variance Distribution Uniformity', fontweight='bold', pad=15)
                ax4.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
                for i, (bar, val) in enumerate(zip(bars4, entropy_data['variance_entropy'])):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Variance Entropy\nData Not Available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_xticks([])
                ax4.set_yticks([])
        else:
            ax4.text(0.5, 0.5, 'Variance Entropy\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # Add consistent styling to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.output_dir / 'efficiency_summary_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _create_feature_dependence_plot(self, data: pd.DataFrame):
        """Create a single clean plot showing HSIC (feature dependence) across methods.

        Lower HSIC = less dependence (better).
        """
        dep_data = data.dropna(subset=['hsic_global_score']).copy()
        if dep_data.empty:
            return
        dep_data = dep_data.sort_values('hsic_global_score', ascending=True)
        methods = dep_data['method_clean']
        scores = dep_data['hsic_global_score'].astype(float).values

        # Colorblind-friendly, scientific colormap
        cmap = plt.get_cmap('cividis')
        # Invert normalization so lower (better) values get brighter/greener colors
        if scores.max() - scores.min() < 1e-12:
            colors = [cmap(0.6)] * len(scores)
        else:
            norm = (scores - scores.min()) / (scores.max() - scores.min())
            # Invert: lower scores (better) get higher colormap values (brighter/greener)
            colors = [cmap(1.0 - v) for v in norm]

        fig, ax = plt.subplots(figsize=(11, 7))
        x = np.arange(len(methods))
        bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.6, alpha=0.95)
        ax.set_xticks(x)
        self.set_colored_xticklabels(ax, methods, rotation=45, ha='right', fontweight='bold')
        # No y-axis label for a cleaner matrix-style presentation
        ax.set_title('Feature Dependence (Lower = Better)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        # no per-bar numeric labels (declutter)
        fig.tight_layout()
        # Ensure long method labels and title are not cut off
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.30, top=0.90)
        plt.savefig(self.output_dir / 'feature_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_feature_structure_quality_plot(self, data: pd.DataFrame):
        """Create a single clean plot for Feature Structure Quality = (1-HSIC) × (active/total)."""
        req = ['hsic_global_score', 'active_units', 'dim']
        if not all(col in data.columns for col in req):
            return
        sq_data = data.dropna(subset=req).copy()
        if sq_data.empty:
            return
        sq_data['independence_score'] = 1 - sq_data['hsic_global_score'].astype(float)
        # Avoid divide-by-zero
        sq_data['efficiency'] = (sq_data['active_units'].astype(float) / np.maximum(sq_data['dim'].astype(float), 1.0))
        sq_data['structure_quality'] = sq_data['independence_score'] * sq_data['efficiency']

        sq_data = sq_data.sort_values('structure_quality', ascending=False)
        methods = sq_data['method_clean']
        vals = sq_data['structure_quality'].values

        cmap = plt.get_cmap('cividis')
        if vals.max() - vals.min() < 1e-12:
            colors = [cmap(0.6)] * len(vals)
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min())
            colors = [cmap(v) for v in norm]

        fig, ax = plt.subplots(figsize=(11, 7))
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.6, alpha=0.95)
        ax.set_xticks(x)
        self.set_colored_xticklabels(ax, methods, rotation=45, ha='right', fontweight='bold')
        # No y-axis label for a cleaner matrix-style presentation
        ax.set_title('Feature Structure Quality (Higher = Better)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        # Add headroom so tallest bars and y-label/title are not clipped
        try:
            vmax = float(np.nanmax(vals))
            if np.isfinite(vmax) and vmax > 0:
                ax.set_ylim(0, vmax * 1.18)
        except Exception:
            pass
        ax.margins(y=0.08)
        fig.tight_layout()
        # Ensure long method labels, y-label, and title are not cut off
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.32, top=0.94)
        plt.savefig(self.output_dir / 'feature_structure_quality.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_information_summary_comparison(self, data: pd.DataFrame):
        """Create information content summary comparison plot."""
        info_data = []
        
        for _, row in data.iterrows():
            row_info = {'method': row['method_clean']}
            
            # Collect MI data for different tasks
            for task in ['gender', 'abnormal']:
                mi_col = f'mi_{task}_mean'
                if mi_col in row and not pd.isna(row[mi_col]):
                    row_info[f'mi_{task}'] = row[mi_col]
            
            if len(row_info) > 1:  # Has at least one MI value
                info_data.append(row_info)
        
        if not info_data:
            return
            
        info_df = pd.DataFrame(info_data).sort_values('method')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Information Content: Method Comparison', fontsize=18, fontweight='bold')
        
        # Plot 1: Gender Task MI
        if 'mi_gender' in info_df.columns:
            gender_data = info_df.dropna(subset=['mi_gender'])
            if not gender_data.empty:
                colors = self.get_professional_colors()
                bars1 = ax1.bar(range(len(gender_data)), gender_data['mi_gender'], 
                              alpha=0.8, color=colors['primary_blue'])
                ax1.set_xticks(range(len(gender_data)))
                self.set_colored_xticklabels(ax1, gender_data['method'], rotation=45, ha='right')
                ax1.set_ylabel('Mutual Information')
                ax1.set_title('Gender Task Informativeness')
                ax1.grid(True, alpha=0.3, axis='y')
                for i, (bar, val) in enumerate(zip(bars1, gender_data['mi_gender'])):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Abnormal Task MI
        if 'mi_abnormal' in info_df.columns:
            abnormal_data = info_df.dropna(subset=['mi_abnormal'])
            if not abnormal_data.empty:
                colors = self.get_professional_colors()
                bars2 = ax2.bar(range(len(abnormal_data)), abnormal_data['mi_abnormal'], 
                              alpha=0.8, color=colors['primary_purple'])
                ax2.set_xticks(range(len(abnormal_data)))
                self.set_colored_xticklabels(ax2, abnormal_data['method'], rotation=45, ha='right')
                ax2.set_ylabel('Mutual Information')
                ax2.set_title('Abnormal Task Informativeness')
                ax2.grid(True, alpha=0.3, axis='y')
                for i, (bar, val) in enumerate(zip(bars2, abnormal_data['mi_abnormal'])):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Average MI across tasks
        if 'mi_gender' in info_df.columns and 'mi_abnormal' in info_df.columns:
            both_tasks = info_df.dropna(subset=['mi_gender', 'mi_abnormal'])
            if not both_tasks.empty:
                colors = self.get_professional_colors()
                avg_mi = (both_tasks['mi_gender'] + both_tasks['mi_abnormal']) / 2
                bars3 = ax3.bar(range(len(both_tasks)), avg_mi, alpha=0.8, color=colors['primary_blue'])
                ax3.set_xticks(range(len(both_tasks)))
                self.set_colored_xticklabels(ax3, both_tasks['method'], rotation=45, ha='right')
                ax3.set_ylabel('Average Mutual Information')
                ax3.set_title('Overall Task Informativeness')
                ax3.grid(True, alpha=0.3, axis='y')
                for i, (bar, val) in enumerate(zip(bars3, avg_mi)):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Side-by-side comparison
        if 'mi_gender' in info_df.columns and 'mi_abnormal' in info_df.columns:
            comparison_data = info_df.dropna(subset=['mi_gender', 'mi_abnormal'])
            if not comparison_data.empty:
                colors = self.get_professional_colors()
                x_pos = np.arange(len(comparison_data))
                width = 0.35
                
                bars4a = ax4.bar(x_pos - width/2, comparison_data['mi_gender'], width,
                               label='Gender Task', alpha=0.8, color=colors['primary_blue'])
                bars4b = ax4.bar(x_pos + width/2, comparison_data['mi_abnormal'], width,
                               label='Abnormal Task', alpha=0.8, color=colors['primary_purple'])
                
                ax4.set_xticks(x_pos)
                self.set_colored_xticklabels(ax4, comparison_data['method'], rotation=45, ha='right')
                ax4.set_ylabel('Mutual Information')
                ax4.set_title('Task-Specific Information Content')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'information_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_independence_summary_comparison(self, data: pd.DataFrame):
        """Create feature independence summary comparison plot."""
        independence_data = data.dropna(subset=['hsic_global_score']).copy()
        if independence_data.empty:
            return
            
        # Convert HSIC to independence score (1 - HSIC, so higher = more independent)
        independence_data['independence_score'] = 1 - independence_data['hsic_global_score']
        # Sort by canonical method order instead of alphabetical
        ordered_methods = self.get_canonical_method_order(independence_data['method_clean'].tolist())
        independence_data = independence_data.set_index('method_clean').reindex(ordered_methods).reset_index()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Independence/Disentanglement: Method Comparison', fontsize=18, fontweight='bold')
        
        methods = independence_data['method_clean']
        x_pos = range(len(methods))
        
        # Plot 1: HSIC Global Score (lower = more independent)
        colors = self.get_professional_colors()
        bars1 = ax1.bar(x_pos, independence_data['hsic_global_score'], alpha=0.8, color=colors['primary_purple'])
        ax1.set_xticks(x_pos)
        self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right')
        ax1.set_ylabel('HSIC Global Score')
        ax1.set_title('Feature Dependence (Lower = Better)')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars1, independence_data['hsic_global_score'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Independence Score (higher = more independent)
        bars2 = ax2.bar(x_pos, independence_data['independence_score'], alpha=0.8, color=colors['primary_blue'])
        ax2.set_xticks(x_pos)
        self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right')
        ax2.set_ylabel('Independence Score (1 - HSIC)')
        ax2.set_title('Feature Independence (Higher = Better)')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars2, independence_data['independence_score'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Independence Ranking
        ranked_data = independence_data.sort_values('independence_score', ascending=False)
        colors3 = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(ranked_data)))
        bars3 = ax3.barh(range(len(ranked_data)), ranked_data['independence_score'], 
                        alpha=0.8, color=colors3)
        ax3.set_yticks(range(len(ranked_data)))
        self.set_colored_yticklabels(ax3, ranked_data['method_clean'])
        ax3.set_xlabel('Independence Score')
        ax3.set_title('Independence Ranking (Worst to Best)')
        ax3.grid(True, alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars3, ranked_data['independence_score'])):
            ax3.text(val + val*0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', ha='left', va='center', fontweight='bold')
        
        # Plot 4: Feature Structure Quality Indicator
        # Create a composite score combining independence with efficiency if available
        if 'active_units' in independence_data.columns and 'dim' in independence_data.columns:
            efficiency = independence_data['active_units'] / independence_data['dim']
            structure_quality = independence_data['independence_score'] * efficiency
            
            bars4 = ax4.bar(x_pos, structure_quality, alpha=0.8, color=colors['primary_purple'])
            ax4.set_xticks(x_pos)
            self.set_colored_xticklabels(ax4, methods, rotation=45, ha='right')
            ax4.set_ylabel('Structure Quality Score')
            ax4.set_title('Feature Structure Quality\n(Independence × Efficiency)')
            ax4.grid(True, alpha=0.3, axis='y')
            for i, (bar, val) in enumerate(zip(bars4, structure_quality)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Structure Quality\nScore Not Available\n(Missing Efficiency Data)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'independence_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_geometric_summary_comparison(self, data: pd.DataFrame):
        """Create geometric preservation summary comparison plot."""
        geometric_data = data.dropna(subset=['trustworthiness', 'continuity', 'dist_corr']).copy()
        if geometric_data.empty:
            return
            
        # Sort by canonical method order instead of alphabetical
        ordered_methods = self.get_canonical_method_order(geometric_data['method_clean'].tolist())
        geometric_data = geometric_data.set_index('method_clean').reindex(ordered_methods).reset_index()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # Removed overall header (suptitle) for cleaner layout
        
        methods = geometric_data['method_clean']
        x_pos = range(len(methods))
        
        # Plot 1: Trustworthiness
        colors = self.get_professional_colors()
        bars1 = ax1.bar(x_pos, geometric_data['trustworthiness'], alpha=0.8, color=colors['primary_blue'])
        ax1.set_xticks(x_pos)
        self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right')
        ax1.set_ylabel('Trustworthiness Score')
        ax1.set_title('Neighborhood Preservation', pad=35)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)
        for i, (bar, val) in enumerate(zip(bars1, geometric_data['trustworthiness'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Continuity
        bars2 = ax2.bar(x_pos, geometric_data['continuity'], alpha=0.8, color=colors['primary_purple'])
        ax2.set_xticks(x_pos)
        self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right')
        ax2.set_ylabel('Continuity Score')
        ax2.set_title('Embedding Smoothness', pad=35)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)
        for i, (bar, val) in enumerate(zip(bars2, geometric_data['continuity'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Distance Correlation
        bars3 = ax3.bar(x_pos, geometric_data['dist_corr'], alpha=0.8, color='coral')
        ax3.set_xticks(x_pos)
        self.set_colored_xticklabels(ax3, methods, rotation=45, ha='right')
        ax3.set_ylabel('Distance Correlation')
        ax3.set_title('Distance Preservation', pad=35)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars3, geometric_data['dist_corr'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Composite Geometric Quality
        composite_score = (geometric_data['trustworthiness'] + 
                         geometric_data['continuity'] + 
                         geometric_data['dist_corr']) / 3
        
        bars4 = ax4.bar(x_pos, composite_score, alpha=0.8, color=colors['primary_blue'])
        ax4.set_xticks(x_pos)
        self.set_colored_xticklabels(ax4, methods, rotation=45, ha='right')
        ax4.set_ylabel('Composite Geometric Score')
        ax4.set_title('Overall Geometric Quality', pad=35)
        ax4.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars4, composite_score)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # Increase top margin so subplot titles don't overlap with bar labels
        fig.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / 'geometric_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_clustering_summary_comparison(self, data: pd.DataFrame):
        """Create cluster quality summary comparison plot."""
        clustering_data = data.dropna(subset=['silhouette', 'davies_bouldin', 'calinski_harabasz']).copy()
        if clustering_data.empty:
            return
            
        # Sort by canonical method order instead of alphabetical
        ordered_methods = self.get_canonical_method_order(clustering_data['method_clean'].tolist())
        clustering_data = clustering_data.set_index('method_clean').reindex(ordered_methods).reset_index()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        
        methods = clustering_data['method_clean']
        x_pos = range(len(methods))
        
        # Plot 1: Silhouette Score
        colors = self.get_professional_colors()
        bars1 = ax1.bar(x_pos, clustering_data['silhouette'], alpha=0.8, color=colors['primary_blue'])
        ax1.set_xticks(x_pos)
        self.set_colored_xticklabels(ax1, methods, rotation=45, ha='right')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Cluster Separation (Higher = Better)', pad=35)
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars1, clustering_data['silhouette'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Davies-Bouldin Index (lower = better; plot raw values)
        bars2 = ax2.bar(x_pos, clustering_data['davies_bouldin'], alpha=0.8, color=colors['primary_purple'])
        ax2.set_xticks(x_pos)
        self.set_colored_xticklabels(ax2, methods, rotation=45, ha='right')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Cluster Compactness (Lower = Better)', pad=35)
        ax2.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars2, clustering_data['davies_bouldin'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Calinski-Harabasz Index
        bars3 = ax3.bar(x_pos, clustering_data['calinski_harabasz'], alpha=0.8, color='coral')
        ax3.set_xticks(x_pos)
        self.set_colored_xticklabels(ax3, methods, rotation=45, ha='right')
        ax3.set_ylabel('Calinski-Harabasz Index')
        ax3.set_title('Variance Ratio (Higher = Better)', pad=35)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars3, clustering_data['calinski_harabasz'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Composite Clustering Quality
        # Normalize all metrics to 0-1 scale for fair comparison
        sil_norm = clustering_data['silhouette']  # Already 0-1
        db_norm = 1 - (clustering_data['davies_bouldin'] / clustering_data['davies_bouldin'].max())  # Keep inverted form only for composite
        ch_norm = clustering_data['calinski_harabasz'] / clustering_data['calinski_harabasz'].max()  # Normalize
        
        composite_score = (sil_norm + db_norm + ch_norm) / 3
        
        bars4 = ax4.bar(x_pos, composite_score, alpha=0.8, color=colors['primary_blue'])
        ax4.set_xticks(x_pos)
        self.set_colored_xticklabels(ax4, methods, rotation=45, ha='right')
        ax4.set_ylabel('Composite Clustering Score')
        ax4.set_title('Overall Clustering Quality', pad=35)
        ax4.grid(True, alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars4, composite_score)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_information_content_analysis(self, data: pd.DataFrame):
        """Create enhanced information content analysis combining HSIC and MI metrics."""
        # Collect information content data
        info_data = []
        
        for _, row in data.iterrows():
            row_data = {'method': row['method_clean']}
            
            # HSIC (independence measure - lower is better)
            if 'hsic_global_score' in row and not pd.isna(row['hsic_global_score']):
                row_data['hsic_global_score'] = row['hsic_global_score']
                row_data['independence_score'] = 1 - row['hsic_global_score']  # Convert to higher-is-better
            
            # MI measures (informativeness - higher is better)
            mi_scores = []
            for task in ['gender', 'abnormal', 'age']:
                mi_col = f'mi_{task}_mean'
                if mi_col in row and not pd.isna(row[mi_col]):
                    row_data[f'mi_{task}'] = row[mi_col]
                    mi_scores.append(row[mi_col])
            
            if mi_scores:
                row_data['mi_average'] = np.mean(mi_scores)
                row_data['mi_total'] = np.sum(mi_scores)
            
            # Information efficiency (high MI with high independence)
            if 'independence_score' in row_data and 'mi_average' in row_data:
                row_data['info_efficiency'] = row_data['independence_score'] * row_data['mi_average']
            
            if len(row_data) > 1:  # Has at least one metric
                info_data.append(row_data)
        
        if info_data:
            info_df = pd.DataFrame(info_data)
            
            # Create comprehensive information analysis plot
            fig = plt.figure(figsize=(24, 18))
            
            # Create subplot layout: 3x3 grid
            gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)
            
            # Plot 1: HSIC Independence Score
            if 'independence_score' in info_df.columns:
                ax1 = fig.add_subplot(gs[0, 0])
                indep_data = info_df.dropna(subset=['independence_score'])
                if not indep_data.empty:
                    sorted_indep = indep_data.sort_values('independence_score', ascending=True)
                    colors1 = plt.cm.RdYlGn(sorted_indep['independence_score'])
                    bars1 = ax1.barh(range(len(sorted_indep)), sorted_indep['independence_score'], 
                                   color=colors1, alpha=0.8)
                    ax1.set_yticks(range(len(sorted_indep)))
                    self.set_colored_yticklabels(ax1, sorted_indep['method'])
                    ax1.set_xlabel('Independence Score (1 - HSIC)')
                    ax1.set_title('Feature Independence\\n(Higher = More Independent)', fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars1, sorted_indep['independence_score'])):
                        ax1.text(val + max(sorted_indep['independence_score'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Average MI Informativeness
            if 'mi_average' in info_df.columns:
                ax2 = fig.add_subplot(gs[0, 1])
                mi_data = info_df.dropna(subset=['mi_average'])
                if not mi_data.empty:
                    sorted_mi = mi_data.sort_values('mi_average', ascending=True)
                    colors2 = plt.cm.plasma(sorted_mi['mi_average'] / sorted_mi['mi_average'].max())
                    bars2 = ax2.barh(range(len(sorted_mi)), sorted_mi['mi_average'], 
                                   color=colors2, alpha=0.8)
                    ax2.set_yticks(range(len(sorted_mi)))
                    self.set_colored_yticklabels(ax2, sorted_mi['method'])
                    ax2.set_xlabel('Average MI Score')
                    ax2.set_title('Feature Informativeness\\n(Higher = More Informative)', fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars2, sorted_mi['mi_average'])):
                        ax2.text(val + max(sorted_mi['mi_average'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Information Efficiency
            if 'info_efficiency' in info_df.columns:
                ax3 = fig.add_subplot(gs[0, 2])
                eff_data = info_df.dropna(subset=['info_efficiency'])
                if not eff_data.empty:
                    sorted_eff = eff_data.sort_values('info_efficiency', ascending=True)
                    colors3 = plt.cm.viridis(sorted_eff['info_efficiency'] / sorted_eff['info_efficiency'].max())
                    bars3 = ax3.barh(range(len(sorted_eff)), sorted_eff['info_efficiency'], 
                                   color=colors3, alpha=0.8)
                    ax3.set_yticks(range(len(sorted_eff)))
                    self.set_colored_yticklabels(ax3, sorted_eff['method'])
                    ax3.set_xlabel('Information Efficiency')
                    ax3.set_title('Information Efficiency\\n(Independence × Informativeness)', fontweight='bold')
                    ax3.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars3, sorted_eff['info_efficiency'])):
                        ax3.text(val + max(sorted_eff['info_efficiency'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Independence vs Informativeness Scatter
            if 'independence_score' in info_df.columns and 'mi_average' in info_df.columns:
                ax4 = fig.add_subplot(gs[1, :2])
                scatter_data = info_df.dropna(subset=['independence_score', 'mi_average'])
                if not scatter_data.empty:
                    # Color by efficiency if available
                    if 'info_efficiency' in scatter_data.columns:
                        colors_scatter = scatter_data['info_efficiency']
                        colormap = 'viridis'
                        color_label = 'Information Efficiency'
                    else:
                        colors_scatter = range(len(scatter_data))
                        colormap = 'Set3'
                        color_label = 'Method Index'
                    
                    scatter = ax4.scatter(scatter_data['independence_score'], scatter_data['mi_average'],
                                        s=200, alpha=0.8, c=colors_scatter, cmap=colormap,
                                        edgecolors='black', linewidth=1)
                    
                    # Add method labels
                    for i, (indep, mi, method) in enumerate(zip(scatter_data['independence_score'],
                                                              scatter_data['mi_average'],
                                                              scatter_data['method'])):
                        ax4.annotate(method, (indep, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=10, fontweight='bold')
                    
                    ax4.set_xlabel('Independence Score (1 - HSIC)')
                    ax4.set_ylabel('Average MI Score')
                    ax4.set_title('Independence vs Informativeness Trade-off', fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    
                    # Add quadrant labels
                    colors = self.get_professional_colors()
                    ax4.axhline(y=np.median(scatter_data['mi_average']), color='red', linestyle='--', alpha=0.5)
                    ax4.axvline(x=np.median(scatter_data['independence_score']), color='red', linestyle='--', alpha=0.5)
                    
                    ax4.text(0.95, 0.95, 'High Independence\\nHigh Informativeness', transform=ax4.transAxes,
                           ha='right', va='top', bbox=dict(boxstyle='round', facecolor=colors['primary_blue'], alpha=0.7))
                    ax4.text(0.05, 0.05, 'Low Independence\\nLow Informativeness', transform=ax4.transAxes,
                           ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor=colors['primary_purple'], alpha=0.7))
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax4)
                    cbar.set_label(color_label, rotation=270, labelpad=20)
            
            # Plot 5: Task-specific MI comparison
            ax5 = fig.add_subplot(gs[1, 2])
            tasks_with_data = []
            for task in ['gender', 'abnormal', 'age']:
                mi_col = f'mi_{task}'
                if mi_col in info_df.columns:
                    tasks_with_data.append(task)
            
            if len(tasks_with_data) >= 2:
                task_data = info_df.dropna(subset=[f'mi_{task}' for task in tasks_with_data])
                if not task_data.empty:
                    x_pos = np.arange(len(task_data))
                    width = 0.8 / len(tasks_with_data)
                    
                    colors_tasks = ['lightblue', 'lightcoral', 'lightgreen']
                    for i, task in enumerate(tasks_with_data):
                        mi_col = f'mi_{task}'
                        bars = ax5.bar(x_pos + i*width, task_data[mi_col], width,
                                     label=f'{task.title()} Task', alpha=0.8, color=colors_tasks[i])
                    
                    ax5.set_xticks(x_pos + width * (len(tasks_with_data) - 1) / 2)
                    self.set_colored_xticklabels(ax5, task_data['method'], rotation=45, ha='right')
                    ax5.set_ylabel('Mutual Information')
                    ax5.set_title('Task-Specific Informativeness', fontweight='bold')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3, axis='y')
            
            # Plot 6: Information content distribution
            if 'mi_total' in info_df.columns:
                ax6 = fig.add_subplot(gs[2, :])
                total_data = info_df.dropna(subset=['mi_total'])
                if not total_data.empty:
                    # Create distribution plot
                    positions = range(len(total_data))
                    
                    # Main bars
                    bars = ax6.bar(positions, total_data['mi_total'], alpha=0.7, color='steelblue')
                    
                    # Add individual task contributions if available
                    bottom_values = np.zeros(len(total_data))
                    colors_stack = ['lightblue', 'lightcoral', 'lightgreen']
                    
                    for i, task in enumerate(['gender', 'abnormal', 'age']):
                        mi_col = f'mi_{task}'
                        if mi_col in total_data.columns:
                            task_values = total_data[mi_col].fillna(0)
                            ax6.bar(positions, task_values, bottom=bottom_values,
                                  alpha=0.8, label=f'{task.title()} Task', color=colors_stack[i])
                            bottom_values += task_values
                    
                    ax6.set_xticks(positions)
                    self.set_colored_xticklabels(ax6, total_data['method'], rotation=45, ha='right')
                    ax6.set_ylabel('Total Mutual Information')
                    ax6.set_title('Total Information Content by Task Contribution', fontweight='bold')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Comprehensive Information Content Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'information_content_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========== PAIRWISE COMPARISON FUNCTIONS ==========
    
    def _center_rows(self, X: np.ndarray) -> np.ndarray:
        """Center rows for HSIC computation."""
        X = np.asarray(X, dtype=np.float64)
        return X - X.mean(axis=0, keepdims=True)

    def _hsic_linear(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear HSIC between two matrices."""
        Xc = self._center_rows(X)
        Yc = self._center_rows(Y)
        K = Xc @ Xc.T  # (n,n)
        L = Yc @ Yc.T  # (n,n)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        return float(np.sum(KH * LH))

    def linear_cka(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Linear CKA similarity between two embeddings (rows = samples)."""
        hsic_xy = self._hsic_linear(Z1, Z2)
        hsic_xx = self._hsic_linear(Z1, Z1)
        hsic_yy = self._hsic_linear(Z2, Z2)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        val = hsic_xy / denom
        return float(np.clip(val, -1.0, 1.0))

    def rbf_cka(self, Z1: np.ndarray, Z2: np.ndarray, gamma1: Optional[float] = None, gamma2: Optional[float] = None) -> float:
        """RBF-CKA using median heuristic for bandwidth if not provided."""
        def _rbf_kernel(Z: np.ndarray, gamma: Optional[float]) -> np.ndarray:
            D2 = pairwise_distances(Z, metric="sqeuclidean")
            if gamma is None:
                nz = D2[D2 > 0]
                if nz.size == 0:
                    gamma_eff = 1.0
                else:
                    gamma_eff = 1.0 / (2.0 * np.median(nz))
            else:
                gamma_eff = gamma
            return np.exp(-gamma_eff * D2)

        K = _rbf_kernel(Z1, gamma1)
        L = _rbf_kernel(Z2, gamma2)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        hsic_xy = float(np.sum(KH * LH))
        hsic_xx = float(np.sum(KH * KH))
        hsic_yy = float(np.sum(LH * LH))
        val = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return float(np.clip(val, -1.0, 1.0))

    def cca_maxcorr(self, Z1: np.ndarray, Z2: np.ndarray, n_components: Optional[int] = None) -> float:
        """Maximum canonical correlation between two embeddings."""
        Z1 = np.asarray(Z1, dtype=np.float64)
        Z2 = np.asarray(Z2, dtype=np.float64)
        n, d1 = Z1.shape
        _, d2 = Z2.shape
        if n < 2 or d1 == 0 or d2 == 0:
            return 0.0
        k = min(d1, d2, n - 1)
        if n_components is not None:
            k = min(k, int(n_components))
        if k < 1:
            return 0.0
        try:
            cca = CCA(n_components=k, max_iter=500)
            Xc = self._center_rows(Z1)
            Yc = self._center_rows(Z2)
            
            # Fit CCA and get canonical correlations directly
            cca.fit(Xc, Yc)
            
            # Transform to get canonical variables
            X_c, Y_c = cca.transform(Xc, Yc)
            
            # Compute correlations between canonical variable pairs
            # (this gives us the actual canonical correlations)
            corrs = []
            for i in range(X_c.shape[1]):
                x = X_c[:, i]
                y = Y_c[:, i]
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    corrs.append(0.0)
                else:
                    corr = np.corrcoef(x, y)[0, 1]
                    if np.isfinite(corr):
                        corrs.append(abs(float(corr)))
                    else:
                        corrs.append(0.0)
            
            return float(max(corrs)) if corrs else 0.0
        except:
            return 0.0

    def distance_geometry_corr(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Correlation of pairwise distances between two spaces."""
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        iu = np.triu_indices_from(D1, k=1)
        a, b = D1[iu], D2[iu]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def knn_jaccard_overlap(self, Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
        """Average Jaccard overlap of k-NN sets between two spaces."""
        n = Z1.shape[0]
        if n <= 1 or k < 1:
            return 0.0
        k = min(k, n - 1)
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        np.fill_diagonal(D1, np.inf)
        np.fill_diagonal(D2, np.inf)
        idx1 = np.argsort(D1, axis=1)[:, :k]
        idx2 = np.argsort(D2, axis=1)[:, :k]
        scores = []
        for i in range(n):
            s1 = set(idx1[i].tolist())
            s2 = set(idx2[i].tolist())
            inter = len(s1 & s2)
            union = len(s1 | s2)
            scores.append(inter / union if union > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    def procrustes_disparity(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Procrustes disparity (lower = more similar) after optimal alignment."""
        try:
            A, B = Z1, Z2
            if A.shape[1] != B.shape[1]:
                # Handle dimension mismatch by padding smaller with zeros
                max_dim = max(A.shape[1], B.shape[1])
                if A.shape[1] < max_dim:
                    A = np.pad(A, ((0, 0), (0, max_dim - A.shape[1])), mode='constant')
                if B.shape[1] < max_dim:
                    B = np.pad(B, ((0, 0), (0, max_dim - B.shape[1])), mode='constant')
            _, _, disparity = procrustes(A, B)
            return float(disparity)
        except Exception:
            return 0.0

    def compute_pairwise_summary(self, Z1: np.ndarray, Z2: np.ndarray, k: int = 10, use_gpu: bool = False) -> Dict[str, float]:
        """Compute comprehensive pairwise similarity metrics."""
        if use_gpu and CUDA_AVAILABLE:
            return self._compute_pairwise_summary_gpu(Z1, Z2, k)
        else:
            results: Dict[str, float] = {
                "cka_linear": self.linear_cka(Z1, Z2),
                "cka_rbf": self.rbf_cka(Z1, Z2),
                "dist_geom_corr": self.distance_geometry_corr(Z1, Z2),
                f"knn_jaccard_k{k}": self.knn_jaccard_overlap(Z1, Z2, k=k),
                "procrustes_disparity": self.procrustes_disparity(Z1, Z2),
            }
            if PAIRWISE_INCLUDE_CCA:
                results["cca_maxcorr"] = self.cca_maxcorr(Z1, Z2)
            return results
    
    def _compute_pairwise_summary_gpu(self, Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> Dict[str, float]:
        """GPU-accelerated version of pairwise similarity computation."""
        device = torch.cuda.current_device()
        
        # Convert to tensors and move to GPU
        Z1_gpu = torch.tensor(Z1, dtype=torch.float32, device=device)
        Z2_gpu = torch.tensor(Z2, dtype=torch.float32, device=device)
        
        results = {}
        
        # GPU-accelerated CKA (most expensive computation)
        results["cka_linear"] = self._linear_cka_gpu(Z1_gpu, Z2_gpu)
        results["cka_rbf"] = self._rbf_cka_gpu(Z1_gpu, Z2_gpu)
        
        # Distance correlation (can be GPU accelerated)
        results["dist_geom_corr"] = self._distance_geometry_corr_gpu(Z1_gpu, Z2_gpu)
        
        # k-NN Jaccard (GPU accelerated)
        results[f"knn_jaccard_k{k}"] = self._knn_jaccard_overlap_gpu(Z1_gpu, Z2_gpu, k)
        
        # For CCA and Procrustes, fall back to CPU (complex decompositions)
        if PAIRWISE_INCLUDE_CCA:
            results["cca_maxcorr"] = self.cca_maxcorr(Z1, Z2)
        results["procrustes_disparity"] = self.procrustes_disparity(Z1, Z2)
        
        return results
    
    def _linear_cka_gpu(self, Z1: torch.Tensor, Z2: torch.Tensor) -> float:
        """GPU-accelerated Linear CKA."""
        # Center the data
        Z1_centered = Z1 - Z1.mean(dim=0, keepdim=True)
        Z2_centered = Z2 - Z2.mean(dim=0, keepdim=True)
        
        # Compute gram matrices
        K = torch.mm(Z1_centered, Z1_centered.t())
        L = torch.mm(Z2_centered, Z2_centered.t())
        
        # Center the gram matrices
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        K_centered = torch.mm(torch.mm(H, K), H)
        L_centered = torch.mm(torch.mm(H, L), H)
        
        # Compute HSIC values
        hsic_xy = torch.sum(K_centered * L_centered)
        hsic_xx = torch.sum(K_centered * K_centered)
        hsic_yy = torch.sum(L_centered * L_centered)
        
        # CKA
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return float(torch.clamp(cka, -1.0, 1.0).cpu())
    
    def _rbf_cka_gpu(self, Z1: torch.Tensor, Z2: torch.Tensor) -> float:
        """GPU-accelerated RBF CKA."""
        # Compute pairwise squared distances
        def pairwise_sq_distances(X):
            X_norm_sq = torch.sum(X**2, dim=1, keepdim=True)
            return X_norm_sq + X_norm_sq.t() - 2 * torch.mm(X, X.t())
        
        D1_sq = pairwise_sq_distances(Z1)
        D2_sq = pairwise_sq_distances(Z2)
        
        # Median heuristic for bandwidth
        gamma1 = 1.0 / (2.0 * torch.median(D1_sq[D1_sq > 0]))
        gamma2 = 1.0 / (2.0 * torch.median(D2_sq[D2_sq > 0]))
        
        # RBF kernels
        K = torch.exp(-gamma1 * D1_sq)
        L = torch.exp(-gamma2 * D2_sq)
        
        # Center kernels
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        K_centered = torch.mm(torch.mm(H, K), H)
        L_centered = torch.mm(torch.mm(H, L), H)
        
        # Compute HSIC values
        hsic_xy = torch.sum(K_centered * L_centered)
        hsic_xx = torch.sum(K_centered * K_centered)
        hsic_yy = torch.sum(L_centered * L_centered)
        
        # CKA
        cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return float(torch.clamp(cka, -1.0, 1.0).cpu())
    
    def _distance_geometry_corr_gpu(self, Z1: torch.Tensor, Z2: torch.Tensor) -> float:
        """GPU-accelerated distance geometry correlation."""
        def pairwise_distances_gpu(X):
            X_norm_sq = torch.sum(X**2, dim=1, keepdim=True)
            return torch.sqrt(X_norm_sq + X_norm_sq.t() - 2 * torch.mm(X, X.t()) + 1e-12)
        
        D1 = pairwise_distances_gpu(Z1)
        D2 = pairwise_distances_gpu(Z2)
        
        # Get upper triangle indices
        n = D1.shape[0]
        triu_indices = torch.triu_indices(n, n, offset=1, device=D1.device)
        
        d1_flat = D1[triu_indices[0], triu_indices[1]]
        d2_flat = D2[triu_indices[0], triu_indices[1]]
        
        # Compute correlation
        if torch.std(d1_flat) < 1e-12 or torch.std(d2_flat) < 1e-12:
            return 0.0
        
        corr = torch.corrcoef(torch.stack([d1_flat, d2_flat]))[0, 1]
        return float(corr.cpu())
    
    def _knn_jaccard_overlap_gpu(self, Z1: torch.Tensor, Z2: torch.Tensor, k: int) -> float:
        """GPU-accelerated k-NN Jaccard overlap."""
        def pairwise_distances_gpu(X):
            X_norm_sq = torch.sum(X**2, dim=1, keepdim=True)
            return torch.sqrt(X_norm_sq + X_norm_sq.t() - 2 * torch.mm(X, X.t()) + 1e-12)
        
        n = Z1.shape[0]
        if n <= 1 or k < 1:
            return 0.0
        k = min(k, n - 1)
        
        D1 = pairwise_distances_gpu(Z1)
        D2 = pairwise_distances_gpu(Z2)
        
        # Set diagonal to infinity
        D1.fill_diagonal_(float('inf'))
        D2.fill_diagonal_(float('inf'))
        
        # Get k nearest neighbors
        _, idx1 = torch.topk(D1, k, dim=1, largest=False)
        _, idx2 = torch.topk(D2, k, dim=1, largest=False)
        
        # Compute Jaccard overlaps
        scores = []
        for i in range(n):
            set1 = set(idx1[i].cpu().numpy())
            set2 = set(idx2[i].cpu().numpy())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            scores.append(intersection / union if union > 0 else 0.0)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _load_latent_features(self, method: str, split: str = 'eval') -> Optional[Tuple[np.ndarray, List[str]]]:
        """Load latent features and sample IDs for a method."""
        cache_key = f"{method}_{split}"
        if cache_key in self.latent_features_cache:
            return self.latent_features_cache[cache_key]
        
        method_dir = self.results_dir / method
        latent_file = method_dir / f"temp_latent_features_{split}.json"
        
        if not latent_file.exists():
            print(f"  ✗ No latent features file for {method} ({split})")
            return None
        
        try:
            # Load JSONL format: [latent_vector, gender, age, abnormal, sample_id]
            latent_vectors = []
            sample_ids = []
            
            with open(latent_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, list) and len(data) >= 5:
                        # Format: [latent_vector, gender, age, abnormal, sample_id]
                        latent_vector = data[0]
                        sample_id = data[4]
                        latent_vectors.append(latent_vector)
                        sample_ids.append(str(sample_id))
                    elif isinstance(data, dict):
                        # Alternative dictionary format
                        if 'latent' in data and 'sample_id' in data:
                            latent_vectors.append(data['latent'])
                            sample_ids.append(str(data['sample_id']))
                        elif 'features' in data and 'sample_id' in data:
                            latent_vectors.append(data['features'])
                            sample_ids.append(str(data['sample_id']))
            
            if latent_vectors and sample_ids:
                Z = np.array(latent_vectors, dtype=np.float32)
                result = (Z, sample_ids)
                self.latent_features_cache[cache_key] = result
                print(f"  ✓ Loaded {method} ({split}): {Z.shape}, {len(sample_ids)} samples")
                return result
            else:
                print(f"  ✗ No latent vectors or sample IDs found in {method} ({split})")
                return None
                
        except Exception as e:
            print(f"  ✗ Error loading {method} ({split}): {e}")
            return None
    
    def _align_latent_features_by_ids(self, Z1: np.ndarray, ids1: List[str], Z2: np.ndarray, ids2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Align two sets of latent features by their sample IDs."""
        if not ids1 or not ids2:
            raise ValueError("Missing sample IDs for alignment.")
        
        # Build index by ID
        idx1_map = {sid: i for i, sid in enumerate(ids1)}
        idx2_map = {sid: i for i, sid in enumerate(ids2)}
        
        # Find common sample IDs
        common_ids = [sid for sid in ids1 if sid in idx2_map]
        
        if not common_ids:
            raise ValueError("No overlapping sample IDs between methods; cannot align.")
        
        if len(common_ids) < min(len(ids1), len(ids2)):
            print(f"    ⚠️  Aligning on {len(common_ids)} common sample IDs (method1={len(ids1)}, method2={len(ids2)})")
        
        # Get aligned indices
        aligned_idx1 = np.array([idx1_map[sid] for sid in common_ids], dtype=int)
        aligned_idx2 = np.array([idx2_map[sid] for sid in common_ids], dtype=int)
        
        return Z1[aligned_idx1], Z2[aligned_idx2]
    
    def compute_pairwise_similarities(self, split: str = 'eval', k: int = 10, use_gpu: bool = True, n_workers: int = None) -> pd.DataFrame:
        """Compute pairwise similarities between all methods in the group."""
        print(f"\nComputing pairwise similarities for {split} split...")
        
        # Load latent features and sample IDs for all available methods
        method_data = {}
        for method in self.methods:
            result = self._load_latent_features(method, split)
            if result is not None:
                Z, sample_ids = result
                method_data[method] = {'features': Z, 'ids': sample_ids}
        
        if len(method_data) < 2:
            print(f"  ✗ Need at least 2 methods with latent features, found {len(method_data)}")
            return pd.DataFrame()
        
        print(f"  ✓ Computing similarities for {len(method_data)} methods")
        
        # Calculate total number of pairs (including diagonal)
        n_methods = len(method_data)
        total_pairs = n_methods * (n_methods + 1) // 2
        print(f"  → Total pairs to compute: {total_pairs} ({n_methods}×{n_methods} upper triangle)")
        
        # Determine processing mode
        if n_workers is None:
            n_workers = min(16, mp.cpu_count())  # Use up to 16 cores
        
        use_gpu_final = use_gpu and CUDA_AVAILABLE
        use_parallel = n_workers > 1 and total_pairs > 10  # Only parallelize for larger workloads
        
        if use_gpu_final:
            print(f"  → Using GPU acceleration (CUDA)")
        elif use_parallel:
            print(f"  → Using CPU parallelization ({n_workers} workers)")
        else:
            print(f"  → Using sequential processing")
        
        methods_list = self.get_canonical_method_order(list(method_data.keys()))  # Use canonical order
        
        # Generate list of pairs to compute
        pairs_to_compute = []
        for i, method1 in enumerate(methods_list):
            for j, method2 in enumerate(methods_list):
                if i <= j:  # Include diagonal and upper triangle
                    pairs_to_compute.append((i, j, method1, method2))
        
        start_time = time.time()
        
        if use_parallel and not use_gpu_final:
            # Parallel CPU processing
            similarity_data = self._compute_similarities_parallel(pairs_to_compute, method_data, k, n_workers)
        else:
            # Sequential processing (GPU or CPU)
            similarity_data = self._compute_similarities_sequential(pairs_to_compute, method_data, k, use_gpu_final)
        
        elapsed_time = time.time() - start_time
        print(f"  ✓ Completed {len(pairs_to_compute)} pairwise comparisons in {elapsed_time:.1f}s ({len(similarity_data)} metric values)")
        return pd.DataFrame(similarity_data)
    
    def _compute_similarities_sequential(self, pairs_to_compute: List[Tuple], method_data: Dict, k: int, use_gpu: bool) -> List[Dict]:
        """Sequential computation of similarities."""
        similarity_data = []
        total_pairs = len(pairs_to_compute)
        
        for pair_idx, (i, j, method1, method2) in enumerate(pairs_to_compute, 1):
            # Progress indication
            method1_clean = self.clean_method_name(method1)
            method2_clean = self.clean_method_name(method2)
            print(f"    [{pair_idx}/{total_pairs}] Computing: {method1_clean} vs {method2_clean}")
            
            data1 = method_data[method1]
            data2 = method_data[method2]
            
            Z1, ids1 = data1['features'], data1['ids']
            Z2, ids2 = data2['features'], data2['ids']
            
            if i == j:
                # Same method - perfect similarity for most metrics
                similarities = {
                    "cka_linear": 1.0,
                    "cka_rbf": 1.0, 
                    "cca_maxcorr": 1.0,
                    "dist_geom_corr": 1.0,
                    f"knn_jaccard_k{k}": 1.0,
                    "procrustes_disparity": 0.0,  # Lower is better
                }
                aligned_samples = len(ids1)
            else:
                # Different methods - need proper alignment by sample IDs
                try:
                    Z1_aligned, Z2_aligned = self._align_latent_features_by_ids(Z1, ids1, Z2, ids2)
                    aligned_samples = Z1_aligned.shape[0]
                    
                    print(f"      → Aligned {aligned_samples} samples, computing similarity metrics...")
                    
                    if aligned_samples < 10:
                        print(f"      ⚠️  Warning: Only {aligned_samples} aligned samples")
                    
                    # Compute actual similarities on aligned data
                    similarities = self.compute_pairwise_summary(Z1_aligned, Z2_aligned, k=k, use_gpu=use_gpu)
                    
                except ValueError as e:
                    print(f"    ✗ Alignment failed for {method1} vs {method2}: {e}")
                    # Skip this pair
                    continue
            
            # Store results
            for metric, value in similarities.items():
                similarity_data.append({
                    'method1': method1,
                    'method2': method2,
                    'method1_clean': self.clean_method_name(method1),
                    'method2_clean': self.clean_method_name(method2),
                    'metric': metric,
                    'value': value,
                    'samples_used': aligned_samples
                })
        
        return similarity_data
    
    def _compute_similarities_parallel(self, pairs_to_compute: List[Tuple], method_data: Dict, k: int, n_workers: int) -> List[Dict]:
        """Parallel computation of similarities using multiprocessing."""
        print(f"    → Launching {n_workers} worker processes...")
        
        # Prepare arguments for the global worker function
        worker_args = [(pair, method_data, k) for pair in pairs_to_compute]
        
        # Execute in parallel
        similarity_data = []
        completed = 0
        total_pairs = len(pairs_to_compute)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs to the global worker function
            future_to_pair = {executor.submit(_global_compute_pair_worker, args): args[0] for args in worker_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_pair):
                pair = future_to_pair[future]
                completed += 1
                
                try:
                    results = future.result()
                    similarity_data.extend(results)
                    
                    if completed % 5 == 0 or completed == total_pairs:  # Progress every 5 pairs
                        print(f"    → Completed {completed}/{total_pairs} pairs")
                        
                except Exception as e:
                    method1, method2 = pair[2], pair[3]
                    print(f"    ✗ Failed to compute {method1} vs {method2}: {e}")
        
        return similarity_data
    
    def create_pairwise_similarity_analysis(self, split: str = 'eval', use_gpu: bool = True, n_workers: int = None):
        """Create comprehensive pairwise similarity analysis and visualizations."""
        print("\n" + "="*60)
        print("PAIRWISE SIMILARITY ANALYSIS")
        print("="*60)
        
        similarity_df = self.compute_pairwise_similarities(split, use_gpu=use_gpu, n_workers=n_workers)
        if similarity_df.empty:
            print("No pairwise similarities computed - skipping analysis")
            return
        
        # Create visualizations for each metric
        metrics = sorted(similarity_df['metric'].unique())  # Ensure consistent order
        
        # Create comprehensive similarity matrices plot
        self._create_similarity_matrices_plot(similarity_df, metrics)
        
        # Create detailed analysis plots
        self._create_similarity_analysis_plots(similarity_df, metrics)
        
        # Save similarity matrices as CSV
        self._save_similarity_matrices(similarity_df, metrics)
        
        print("  ✓ Saved pairwise similarity analysis")
    
    def _create_similarity_matrices_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create comprehensive similarity matrices visualization with hierarchical clustering."""
        # Determine grid size (3 rows x 2 cols instead of 2x3)
        n_metrics = len(metrics)
        rows = 2
        cols = 3
        
        fig = plt.figure(figsize=(6*cols, 5*rows))
        
        for idx, metric in enumerate(metrics):
            ax = plt.subplot(rows, cols, idx + 1)
            
            # Create pivot table for this metric
            metric_data = similarity_df[similarity_df['metric'] == metric]
            pivot_matrix = metric_data.pivot(index='method1_clean', columns='method2_clean', values='value')
            
            # Make symmetric by filling lower triangle
            pivot_matrix = pivot_matrix.fillna(pivot_matrix.T)
            
            # Compute individual hierarchical ordering for this specific matrix
            if not pivot_matrix.empty:
                row_order, col_order = self._compute_hierarchical_method_order_for_matrix(pivot_matrix)
                try:
                    pivot_matrix = pivot_matrix.reindex(index=row_order, columns=col_order)
                except:
                    # Keep original order if reordering fails
                    pass
            
            # Choose appropriate colormap and bounds based on metric
            if 'disparity' in metric:
                # Lower is better (>=0)
                cmap = 'Reds_r'
                vmin, vmax = 0, float(pivot_matrix.max().max())
                center = None
            elif 'dist_geom_corr' in metric:
                # Pearson correlation in [-1, 1]
                cmap = 'RdBu_r'
                vmin, vmax = -1, 1
                center = 0
            else:
                # Similarity scores in [0, 1]
                cmap = 'RdYlGn'
                vmin, vmax = 0, 1
                center = None
            
            # Create heatmap with scientific styling
            sns.heatmap(pivot_matrix, annot=False, cmap=cmap, fmt='.3f',
                       square=True, ax=ax, vmin=vmin, vmax=vmax, center=center, 
                       linewidths=0.8, linecolor='white',
                       cbar_kws={"shrink": .8, "aspect": 20, "pad": 0.02})
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', pad=15)
            # Remove axis descriptors for cleaner look
            ax.set_xlabel('')
            ax.set_ylabel('')
            # Rotate labels for better readability with scientific styling and color coding
            # Use the actual pivot_matrix order (after reindexing)
            current_row_methods = pivot_matrix.index.tolist()
            current_col_methods = pivot_matrix.columns.tolist()
            
            # Set colored tick labels using the current order
            self.set_colored_xticklabels(ax, current_col_methods, rotation=45, ha='right', fontsize=9)
            self.set_colored_yticklabels(ax, current_row_methods, rotation=0, fontsize=9)
            
            # Add subtle border
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
        
        # Remove overarching title
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_matrices.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _compute_hierarchical_method_order_for_matrix(self, matrix: pd.DataFrame):
        """Compute optimal method ordering for a single similarity matrix using hierarchical clustering."""
        try:
            if matrix.empty or matrix.shape[0] < 2:
                return matrix.index.tolist()
            
            # Use seaborn's clustermap to get optimal ordering
            # This automatically handles the clustering and reordering
            temp_clustermap = sns.clustermap(matrix, method='ward', metric='euclidean',
                                           figsize=(1, 1), cbar=False, 
                                           xticklabels=False, yticklabels=False)
            plt.close(temp_clustermap.fig)  # Close immediately to avoid display
            
            # Extract the reordered indices
            row_order = temp_clustermap.dendrogram_row.reordered_ind
            col_order = temp_clustermap.dendrogram_col.reordered_ind
            
            # Get the reordered method names
            ordered_methods_row = [matrix.index[i] for i in row_order]
            ordered_methods_col = [matrix.columns[i] for i in col_order]
            
            # For symmetric matrices, use the same ordering for both
            return ordered_methods_row, ordered_methods_col
            
        except Exception as e:
            print(f"Warning: Could not compute hierarchical ordering for matrix: {e}")
            # Fallback to original order
            return matrix.index.tolist(), matrix.columns.tolist()
    
    def _create_similarity_analysis_plots(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create detailed similarity analysis plots."""
        # Plot 1: Average similarity rankings
        self._create_similarity_rankings_plot(similarity_df, metrics)
        
        # Plot 2: Similarity correlations between metrics
        self._create_similarity_correlations_plot(similarity_df, metrics)
        
        # Skip similarity distributions - not necessary for evaluation
        # self._create_similarity_distributions_plot(similarity_df, metrics)
    
    def _create_similarity_rankings_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create rankings based on average similarity to other methods."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            metric_data = similarity_df[similarity_df['metric'] == metric]
            
            # Calculate average similarity for each method (excluding self-comparison)
            avg_similarities = []
            methods = self.get_canonical_method_order(metric_data['method1_clean'].unique())  # Use canonical order
            
            for method in methods:
                # Get similarities where this method is involved (excluding self)
                method_sims = metric_data[
                    ((metric_data['method1_clean'] == method) | (metric_data['method2_clean'] == method)) &
                    (metric_data['method1_clean'] != metric_data['method2_clean'])
                ]['value'].values
                
                if len(method_sims) > 0:
                    if 'disparity' in metric:
                        # Lower is better - use negative for consistent ranking
                        avg_sim = -np.mean(method_sims)
                    else:
                        avg_sim = np.mean(method_sims)
                    avg_similarities.append((method, avg_sim))
            
            # Sort by score but maintain alphabetical order for ties
            avg_similarities.sort(key=lambda x: (-x[1], x[0]))  # Sort by score desc, then name asc
            methods_sorted, scores_sorted = zip(*avg_similarities)
            
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(methods_sorted)))
            bars = ax.barh(range(len(methods_sorted)), scores_sorted, color=colors, alpha=0.8)
            
            ax.set_yticks(range(len(methods_sorted)))
            self.set_colored_yticklabels(ax, methods_sorted)
            ax.set_xlabel(f'Average {metric.replace("_", " ").title()}')
            ax.set_title(f'Method Rankings by {metric.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
                ax.text(score + max(scores_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{abs(score):.3f}', ha='left', va='center', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Method Rankings by Similarity Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_similarity_correlations_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create correlation analysis between different similarity metrics."""
        # Create correlation matrix between metrics
        
        # Get all unique method pairs (excluding self-comparisons)
        pairs_data = similarity_df[similarity_df['method1_clean'] != similarity_df['method2_clean']]
        
        if pairs_data.empty:
            return
        
        # Pivot to get metrics as columns
        pivot_data = pairs_data.pivot_table(
            index=['method1_clean', 'method2_clean'], 
            columns='metric', 
            values='value'
        ).reset_index()
        
        # Calculate correlation matrix
        metric_cols = [col for col in pivot_data.columns if col in metrics]
        if len(metric_cols) < 2:
            return
            
        corr_matrix = pivot_data[metric_cols].corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Similarity Metrics Correlation Matrix', fontweight='bold')
        
        # Plot 2: Pairwise scatter plot (best correlated pair)
        if len(metric_cols) >= 2:
            # Find best correlated pair (excluding diagonal)
            corr_abs = np.abs(corr_matrix.values)
            np.fill_diagonal(corr_abs, 0)
            i, j = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
            
            metric1, metric2 = metric_cols[i], metric_cols[j]
            corr_val = corr_matrix.iloc[i, j]
            
            scatter_data = pivot_data.dropna(subset=[metric1, metric2])
            if not scatter_data.empty:
                ax2.scatter(scatter_data[metric1], scatter_data[metric2], 
                           alpha=0.7, s=100, color='darkblue')
                
                # Add trend line
                z = np.polyfit(scatter_data[metric1], scatter_data[metric2], 1)
                p = np.poly1d(z)
                ax2.plot(scatter_data[metric1], p(scatter_data[metric1]), 
                        "r--", alpha=0.8, linewidth=2)
                
                ax2.set_xlabel(metric1.replace('_', ' ').title())
                ax2.set_ylabel(metric2.replace('_', ' ').title())
                ax2.set_title(f'{metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}\\nr = {corr_val:.3f}', 
                             fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_similarity_distributions_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create distribution analysis of similarity values."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            metric_data = similarity_df[
                (similarity_df['metric'] == metric) & 
                (similarity_df['method1_clean'] != similarity_df['method2_clean'])
            ]
            
            if not metric_data.empty:
                colors = self.get_professional_colors()
                values = metric_data['value'].values
                
                # Create histogram
                ax.hist(values, bins=20, alpha=0.7, color=colors['primary_blue'], edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(values):.3f}')
                ax.axvline(np.median(values), color=colors['primary_blue'], linestyle='--', linewidth=2,
                          label=f'Median: {np.median(values):.3f}')
                
                ax.set_xlabel(f'{metric.replace("_", " ").title()} Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Similarity Metrics Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_similarity_matrices(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Save similarity matrices as CSV files."""
        for metric in metrics:
            metric_data = similarity_df[similarity_df['metric'] == metric]
            pivot_matrix = metric_data.pivot(index='method1_clean', columns='method2_clean', values='value')
            # Make symmetric
            pivot_matrix = pivot_matrix.fillna(pivot_matrix.T)
            
            filename = f'pairwise_{metric}_matrix.csv'
            pivot_matrix.to_csv(self.output_dir / filename)
        
        print(f"  ✓ Saved {len(metrics)} similarity matrices as CSV files")
    
    def create_dimensionality_comparison_plots(self, latent_df: pd.DataFrame, pca_df: pd.DataFrame):
        """Create individual dimensionality comparison plots."""
        eval_data = latent_df[latent_df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        # Plot 1: Stacked bar chart for dimensions
        methods_with_dim = eval_data.dropna(subset=['dim', 'active_units'])
        if not methods_with_dim.empty:
            self._create_stacked_dimensions_plot(methods_with_dim)
        
        # Skip efficiency scatter plot - not needed
        # if not methods_with_dim.empty:
        #     self._create_efficiency_scatter_plot(methods_with_dim)
        
        # Plot 3: PCA effective dimensionality (if available)
        if not pca_df.empty:
            pca_df['method_clean'] = pca_df['method'].apply(self.clean_method_name)
            self._create_pca_dimensions_plot(pca_df)
        
        print("  ✓ Saved dimensionality comparison plots")
    
    def _create_stacked_dimensions_plot(self, data: pd.DataFrame):
        """Create stacked bar chart for total vs active dimensions."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = self.get_professional_colors()
        inactive_units = data['dim'] - data['active_units']
        
        # Create stacked bars
        p1 = ax.bar(range(len(data)), data['active_units'], 
                   label='Active Units', alpha=0.8, color=colors['primary_blue'])
        p2 = ax.bar(range(len(data)), inactive_units, 
                   bottom=data['active_units'], label='Inactive Units', 
                   alpha=0.8, color=colors['primary_purple'])
        
        ax.set_xticks(range(len(data)))
        self.set_colored_xticklabels(ax, data['method_clean'], rotation=45, ha='right')
        ax.set_ylabel('Number of Dimensions')
        ax.set_title('Active vs Inactive Dimensions by Method', fontsize=16, fontweight='bold')
        ax.legend()
        
        # Add efficiency percentages on top
        for i, (active, total) in enumerate(zip(data['active_units'], data['dim'])):
            efficiency = (active / total) * 100
            ax.text(i, total + total*0.01, f'{efficiency:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensions_stacked.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_efficiency_scatter_plot(self, data: pd.DataFrame):
        """Create scatter plot showing dimensionality efficiency."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        efficiency = (data['active_units'] / data['dim']) * 100
        total_dims = data['dim']
        
        # Create scatter plot with size based on active units
        scatter = ax.scatter(total_dims, efficiency, 
                           s=data['active_units']*2, alpha=0.7, 
                           c=range(len(data)), cmap='viridis')
        
        # Add method labels
        for i, (_, row) in enumerate(data.iterrows()):
            eff = (row['active_units'] / row['dim']) * 100
            ax.annotate(row['method_clean'], 
                       (row['dim'] + 5, eff + 1),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Total Dimensions')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Dimensionality Efficiency vs Total Dimensions\n(Bubble size = Active Units)', 
                    fontsize=16, fontweight='bold')
        
        # Add reference lines
        colors = self.get_professional_colors()
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax.axhline(y=50, color=colors['primary_blue'], linestyle='--', alpha=0.5, label='50% Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pca_dimensions_plot(self, pca_df: pd.DataFrame):
        """Create PCA effective dimensionality comparison plot."""
        methods_with_pca = pca_df.dropna(subset=['effective_dim_95', 'effective_dim_99'])
        if not methods_with_pca.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(methods_with_pca))
            width = 0.35
            
            # Create grouped bars
            bars1 = ax.bar(x_pos - width/2, methods_with_pca['effective_dim_95'], 
                          width, label='95% Variance', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x_pos + width/2, methods_with_pca['effective_dim_99'], 
                          width, label='99% Variance', alpha=0.8, color='lightsteelblue')
            
            ax.set_xticks(x_pos)
            self.set_colored_xticklabels(ax, methods_with_pca['method_clean'], rotation=45, ha='right')
            ax.set_ylabel('Effective Dimensions')
            ax.set_title('PCA Effective Dimensionality Comparison', fontsize=16, fontweight='bold')
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'pca_dimensions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_mutual_information_plots(self, df: pd.DataFrame):
        """Create individual mutual information comparison plots."""
        eval_data = df[df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        tasks = ['gender', 'abnormal']
        for task in tasks:
            mi_col = f'mi_{task}_mean'
            data_to_plot = eval_data.dropna(subset=[mi_col])
            if not data_to_plot.empty:
                self._create_mi_plot(data_to_plot, task, mi_col)
        
        # Create combined comparison plot
        self._create_mi_comparison_plot(eval_data)
        
        print("  ✓ Saved mutual information plots")
    
    def _create_mi_plot(self, data: pd.DataFrame, task: str, mi_col: str):
        """Create individual MI plot for a specific task."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by MI value for better visualization
        sorted_data = data.sort_values(mi_col)
        y_pos = range(len(sorted_data))
        
        # Create horizontal lollipop plot
        colors = plt.cm.plasma(sorted_data[mi_col] / sorted_data[mi_col].max())
        
        # Draw stems
        ax.hlines(y_pos, 0, sorted_data[mi_col], colors='gray', alpha=0.4, linewidth=2)
        # Draw circles
        ax.scatter(sorted_data[mi_col], y_pos, color=colors, s=120, alpha=0.9, zorder=5)
        
        ax.set_yticks(y_pos)
        self.set_colored_yticklabels(ax, sorted_data['method_clean'])
        ax.set_xlabel('Mean Mutual Information')
        ax.set_title(f'Feature Informativeness for {task.title()} Task', 
                    fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, (val, method) in enumerate(zip(sorted_data[mi_col], sorted_data['method_clean'])):
            ax.text(val + max(sorted_data[mi_col])*0.01, i, f'{val:.4f}', 
                   ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mutual_information_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_mi_comparison_plot(self, data: pd.DataFrame):
        """Create side-by-side MI comparison for both tasks."""
        mi_data = []
        for _, row in data.iterrows():
            if not pd.isna(row.get('mi_gender_mean')) and not pd.isna(row.get('mi_abnormal_mean')):
                mi_data.append({
                    'method': row['method_clean'],
                    'gender': row['mi_gender_mean'],
                    'abnormal': row['mi_abnormal_mean']
                })
        
        if mi_data:
            mi_df = pd.DataFrame(mi_data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = self.get_professional_colors()
            x_pos = np.arange(len(mi_df))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, mi_df['gender'], width, 
                          label='Gender Task', alpha=0.8, color=colors['primary_blue'])
            bars2 = ax.bar(x_pos + width/2, mi_df['abnormal'], width, 
                          label='Abnormal Task', alpha=0.8, color=colors['primary_purple'])
            
            ax.set_xticks(x_pos)
            self.set_colored_xticklabels(ax, mi_df['method'], rotation=45, ha='right')
            ax.set_ylabel('Mean Mutual Information')
            ax.set_title('Feature Informativeness Comparison Across Tasks', 
                        fontsize=16, fontweight='bold')
            ax.legend()
            
            # Removed value labels on bars (declutters figure)
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mutual_information_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_tables(self, class_df: pd.DataFrame, latent_df: pd.DataFrame, pca_df: pd.DataFrame):
        """Create comprehensive summary tables."""
        
        # Table 1: Best performance per metric
        print("\nCreating summary tables...")
        
        # Classification performance summary
        class_summary = []
        for task in ['gender', 'abnormal']:
            task_data = class_df[class_df['task'] == task]
            for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
                if metric in task_data.columns:
                    best_row = task_data.loc[task_data[metric].idxmax()]
                    class_summary.append({
                        'Task': task.title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Best Method': self.clean_method_name(best_row['method']),
                        'Value': f"{best_row[metric]:.4f}"
                    })
        
        class_summary_df = pd.DataFrame(class_summary)
        class_summary_df.to_csv(self.output_dir / 'classification_summary.csv', index=False)
        print("  ✓ Saved classification_summary.csv")
        
        # Latent space summary
        eval_latent = latent_df[latent_df['split'] == 'eval']
        latent_summary = []
        
        # Core latent metrics
        core_metrics = ['hsic_global_score', 'active_units', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
        
        # Enhanced metrics
        enhanced_metrics = [
            'variance_entropy', 'effective_dim_variance', 'variance_concentration',
            'mi_gender_mean', 'mi_abnormal_mean', 'mi_gender_concentration', 'mi_abnormal_concentration'
        ]
        
        all_metrics = core_metrics + enhanced_metrics
        
        for metric in all_metrics:
            if metric in eval_latent.columns:
                data_available = eval_latent.dropna(subset=[metric])
                if not data_available.empty:
                    # Handle metrics with different optimization directions
                    if metric in ['hsic_global_score', 'davies_bouldin']:  # Lower is better
                        best_row = data_available.loc[data_available[metric].idxmin()]
                        direction = "(Lower Better)"
                    else:  # Higher is better
                        best_row = data_available.loc[data_available[metric].idxmax()]
                        if 'concentration' in metric:
                            direction = "(Higher = More Concentrated)"
                        else:
                            direction = "(Higher Better)"
                    
                    latent_summary.append({
                        'Metric': metric.replace('_', ' ').title() + f" {direction}",
                        'Best Method': self.clean_method_name(best_row['method']),
                        'Value': f"{best_row[metric]:.4f}",
                        'Methods Compared': len(data_available)
                    })
        
        latent_summary_df = pd.DataFrame(latent_summary)
        latent_summary_df.to_csv(self.output_dir / 'latent_summary.csv', index=False)
        print("  ✓ Saved latent_summary.csv")
        
        # Complete comparison table
        methods_list = list(self.available_methods.keys())
        complete_table = []
        
        for method in methods_list:
            row = {'Method': self.clean_method_name(method)}
            
            # Classification metrics
            for task in ['gender', 'abnormal']:
                task_data = class_df[(class_df['task'] == task) & (class_df['method'] == method)]
                if not task_data.empty:
                    for metric in ['accuracy', 'f1', 'roc_auc']:
                        row[f'{task}_{metric}'] = f"{task_data.iloc[0][metric]:.3f}" if not pd.isna(task_data.iloc[0][metric]) else "N/A"
                else:
                    for metric in ['accuracy', 'f1', 'roc_auc']:
                        row[f'{task}_{metric}'] = "N/A"
            
            # Latent metrics (core + key enhanced metrics)
            latent_data = eval_latent[eval_latent['method'] == method]
            if not latent_data.empty:
                # Core metrics
                core_latent_metrics = ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
                # Enhanced metrics
                enhanced_latent_metrics = ['variance_entropy', 'mi_gender_mean', 'mi_abnormal_mean']
                
                all_latent_metrics = core_latent_metrics + enhanced_latent_metrics
                
                for metric in all_latent_metrics:
                    val = latent_data.iloc[0][metric] if metric in latent_data.iloc[0] else None
                    if metric in ['dim', 'active_units']:
                        row[metric] = f"{int(val)}" if not pd.isna(val) else "N/A"
                    else:
                        row[metric] = f"{val:.3f}" if not pd.isna(val) else "N/A"
                
                # Add efficiency calculation
                if not pd.isna(latent_data.iloc[0].get('active_units')) and not pd.isna(latent_data.iloc[0].get('dim')):
                    efficiency = latent_data.iloc[0]['active_units'] / latent_data.iloc[0]['dim']
                    row['efficiency'] = f"{efficiency:.3f}"
                else:
                    row['efficiency'] = "N/A"
            else:
                for metric in ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr',
                              'variance_entropy', 'mi_gender_mean', 'mi_abnormal_mean', 'efficiency']:
                    row[metric] = "N/A"
            
            complete_table.append(row)
        
        # Add missing methods as rows with all N/A
        for method in self.missing_methods:
            row = {'Method': self.clean_method_name(method)}
            for col in complete_table[0].keys():
                if col != 'Method':
                    row[col] = "N/A"
            complete_table.append(row)
        
        complete_df = pd.DataFrame(complete_table)
        complete_df.to_csv(self.output_dir / 'complete_comparison_table.csv', index=False)
        print("  ✓ Saved complete_comparison_table.csv")
        
        return class_summary_df, latent_summary_df, complete_df
    
    def create_correlation_heatmap(self, latent_df: pd.DataFrame):
        """Skip correlation heatmap as it's not interpretable for evaluation."""
        # Removed - correlation matrices between metrics are not useful for evaluation
        print("  ⚬ Skipped metrics correlation heatmap (not interpretable)")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive comparison report with all plots and tables."""
        print("="*60)
        print("COMPREHENSIVE METRICS COMPARISON REPORT")
        print("="*60)
        
        # Load all metrics
        self.load_metrics()
        
        if len(self.available_methods) == 0:
            print("ERROR: No methods with available metrics found!")
            return
        
        print(f"\nAnalyzing {len(self.available_methods)} methods with complete metrics...")
        print(f"Methods: {list(self.available_methods.keys())}")
        
        # Extract different types of metrics
        print("\nExtracting metrics...")
        class_df = self.extract_classification_metrics()
        latent_df = self.extract_latent_metrics()
        pca_df = self.extract_pca_metrics()
        
        print(f"  ✓ Classification metrics: {len(class_df)} records")
        print(f"  ✓ Latent space metrics: {len(latent_df)} records")
        print(f"  ✓ PCA metrics: {len(pca_df)} records")
        
        # Create visualizations
        print("\nGenerating individual comparison plots...")
        
        if not class_df.empty:
            self.create_classification_comparison_plots(class_df)
            # Create classification performance summary comparison
            self._create_classification_summary_comparison(class_df)
        
        if not latent_df.empty:
            self.create_latent_comparison_plots(latent_df)
            self.create_mutual_information_plots(latent_df)
            self.create_correlation_heatmap(latent_df)
        
        if not latent_df.empty or not pca_df.empty:
            self.create_dimensionality_comparison_plots(latent_df, pca_df)
        
        # Create pairwise similarity analysis with acceleration
        if PAIRWISE_AUTO:
            # Automatic mode selection
            self.create_pairwise_similarity_analysis()
        else:
            # Manual configuration
            self.create_pairwise_similarity_analysis(use_gpu=PAIRWISE_USE_GPU, n_workers=PAIRWISE_N_WORKERS)
        
        # Create summary tables
        if not class_df.empty and not latent_df.empty:
            class_summary, latent_summary, complete_table = self.create_summary_tables(class_df, latent_df, pca_df)
            
            print("\n" + "="*60)
            print("SUMMARY RESULTS")
            print("="*60)
            
            print("\nBest Classification Performance:")
            print(class_summary.to_string(index=False))
            
            print("\nBest Latent Space Quality:")
            print(latent_summary.to_string(index=False))
            
            print(f"\nMethods with missing metrics ({len(self.missing_methods)}):")
            for method in self.missing_methods:
                print(f"  - {self.clean_method_name(method)}")
        
        print(f"\n🎉 Analysis complete! All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            if file.name != "metrics_and_plots.py":
                print(f"  - {file.name}")


def main():
    """Main function to run the comprehensive metrics comparison."""
    
  
    small_aggregated = ["tuh-ctm_cma_avg", "tuh-ctm_nn_avg", "tuh-hopf_avg", "tuh-jr_avg", "tuh-wong_wang_avg", "tuh-pca_avg", "tuh-psd_ae_avg"]

   
    medium_unrestricted = ["tuh-ctm_nn_pc","tuh-hopf_pc", "tuh-jr_pc", "tuh-c22", "tuh-pca_pc", "tuh-psd_ae_pc",  "tuh-eegnet"]



    method_groups= [{"small_aggregated": small_aggregated}, {"medium_unrestricted": medium_unrestricted}]



    # =================================================================
    for method_group in method_groups:
        # Initialize and run the comparison  
        comparison = MetricsComparison(method_group=method_group)
        # Delete the output directory for the current group before running the comparison
        if comparison.output_dir.exists() and comparison.output_dir.is_dir():
            shutil.rmtree(comparison.output_dir)
            print(f"Deleted output directory for method group: {comparison.output_dir}")
            comparison.output_dir.mkdir(exist_ok=True)
        comparison.generate_comprehensive_report()


if __name__ == "__main__":
    main()
