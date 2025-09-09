#!/usr/bin/env python3
"""
Recompute geometry metrics (trustworthiness, continuity, distance correlation) 
using EEG PSD feature space as the reference instead of PCA(2).

This script:
1. Loads existing latent features and sample IDs from all methods
2. Loads original EEG data and computes PSDs using standard parameters
3. Matches samples by ID between latent and PSD spaces
4. Recomputes geometry metrics comparing latent space to PSD space
5. Updates the final metrics and saves new results
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import mne
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary utilities
import sys
sys.path.append('/rds/general/user/lrh24/home/thesis/code')
from utils.util import compute_psd_from_raw, PSD_CALCULATION_PARAMS, STANDARD_EEG_CHANNELS
from data_preprocessing import data_loading as dl

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


def _trustworthiness(X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 10, max_samples: int = 5000) -> float:
    """Compute trustworthiness metric measuring how well local neighborhoods are preserved in dimensionality reduction."""
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
    """Compute continuity metric measuring preservation of neighborhood relationships from high to low dimensions."""
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
    """Compute correlation between pairwise distances in high and low dimensional spaces."""
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


def load_latent_features(method: str, results_dir: Path, split: str = 'eval') -> Optional[Tuple[np.ndarray, List[str]]]:
    """Load pre-computed latent features and corresponding sample identifiers from extraction results."""
    method_dir = results_dir / method
    
    # Try different possible filenames
    possible_files = [
        method_dir / f"temp_latent_features_{split}.json",
        method_dir / f"latent_features_{split}.json",
        method_dir / f"features_{split}.json"
    ]
    
    for latent_file in possible_files:
        if latent_file.exists():
            try:
                print(f"  Loading {method} ({split}) from {latent_file}")
                latent_vectors = []
                sample_ids = []
                
                with open(latent_file, 'r') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if isinstance(data, list) and len(data) >= 5:
                            # Format: [latent_vector, gender, age, abnormal, sample_id]
                            latent_vectors.append(np.array(data[0], dtype=np.float32))
                            sample_id = data[4]
                            sample_ids.append(str(sample_id))
                        elif isinstance(data, dict):
                            if 'latent' in data and 'sample_id' in data:
                                latent_vectors.append(np.array(data['latent'], dtype=np.float32))
                                sample_ids.append(str(data['sample_id']))
                            elif 'features' in data and 'sample_id' in data:
                                latent_vectors.append(np.array(data['features'], dtype=np.float32))
                                sample_ids.append(str(data['sample_id']))
                
                if latent_vectors and sample_ids:
                    Z = np.array(latent_vectors, dtype=np.float32)
                    print(f"    ✓ Loaded {method} ({split}): {Z.shape}, {len(sample_ids)} samples")
                    return Z, sample_ids
                    
            except Exception as e:
                print(f"    ✗ Error loading {latent_file}: {e}")
                continue
    
    print(f"    ✗ Could not load latent features for {method} ({split})")
    return None


def load_and_compute_psd_features(data_path: str, split: str = 'eval', 
                                 max_samples: Optional[int] = None, 
                                 use_average: bool = False) -> Tuple[np.ndarray, List[str]]:
    """Load raw EEG data and compute standardized PSD features for geometry metric computation."""
    print(f"  Loading EEG data for {split} split...")
    
    # Load the raw EEG data
    eeg_data = dl.load_data(data_path, split)
    print(f"    Loaded {len(eeg_data)} samples from {split} split")
    
    # Limit samples if specified (for testing/debugging)
    if max_samples and len(eeg_data) > max_samples:
        print(f"    Limiting to first {max_samples} samples for testing")
        eeg_data = eeg_data[:max_samples]
    
    psd_features = []
    sample_ids = []
    
    for i, (raw, gender, age, abnormal, sample_id) in enumerate(eeg_data):
        if i % 100 == 0:
            print(f"    Processing sample {i+1}/{len(eeg_data)}")
        
        try:
            # Compute PSD using the same parameters as the pipeline
            psd = compute_psd_from_raw(
                raw, 
                calculate_average=use_average,  # Average for _avg methods, per-channel for others
                normalize=True,
                n_fft=PSD_CALCULATION_PARAMS["n_fft"],
                n_overlap=PSD_CALCULATION_PARAMS["n_overlap"], 
                n_per_seg=PSD_CALCULATION_PARAMS["n_per_seg"]
            )
            
            # Flatten to create feature vector
            if use_average:
                psd_flat = psd.astype(np.float32)  # Already 1D (F,)
            else:
                psd_flat = psd.flatten().astype(np.float32)  # Flatten (C, F) -> (C*F,)
            psd_features.append(psd_flat)
            sample_ids.append(str(sample_id))
            
        except Exception as e:
            print(f"    ✗ Error processing sample {sample_id}: {e}")
            continue
    
    if psd_features:
        psd_matrix = np.array(psd_features, dtype=np.float32)
        print(f"    ✓ Computed PSDs: {psd_matrix.shape}, {len(sample_ids)} samples")
        return psd_matrix, sample_ids
    else:
        raise ValueError("No PSD features could be computed")


def align_features_by_sample_id(latent_features: np.ndarray, latent_ids: List[str],
                               psd_features: np.ndarray, psd_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align latent and PSD features by sample ID."""
    
    # Create ID mappings
    latent_id_map = {sid: i for i, sid in enumerate(latent_ids)}
    psd_id_map = {sid: i for i, sid in enumerate(psd_ids)}
    
    # Find common sample IDs
    common_ids = [sid for sid in latent_ids if sid in psd_id_map]
    
    if not common_ids:
        raise ValueError("No common sample IDs found between latent and PSD features")
    
    print(f"    Found {len(common_ids)} common samples out of {len(latent_ids)} latent and {len(psd_ids)} PSD")
    
    # Align features
    latent_indices = [latent_id_map[sid] for sid in common_ids]
    psd_indices = [psd_id_map[sid] for sid in common_ids]
    
    aligned_latent = latent_features[latent_indices]
    aligned_psd = psd_features[psd_indices]
    
    return aligned_latent, aligned_psd, common_ids


def compute_geometry_metrics(latent_features: np.ndarray, psd_features: np.ndarray, 
                           max_samples: int = 5000) -> Dict[str, float]:
    """Compute geometry preservation metrics between latent and PSD spaces."""
    
    print(f"    Computing geometry metrics on {latent_features.shape[0]} aligned samples...")
    
    # Compute the three metrics
    trustworthiness = _trustworthiness(psd_features, latent_features, 
                                     n_neighbors=10, max_samples=max_samples)
    continuity = _continuity(psd_features, latent_features, 
                           n_neighbors=10, max_samples=max_samples)
    dist_corr = _distance_correlation(psd_features, latent_features, 
                                    max_samples=max_samples)
    
    metrics = {
        'trustworthiness': float(trustworthiness),
        'continuity': float(continuity), 
        'dist_corr': float(dist_corr)
    }
    
    print(f"      Trustworthiness: {metrics['trustworthiness']:.3f}")
    print(f"      Continuity: {metrics['continuity']:.3f}")
    print(f"      Distance Correlation: {metrics['dist_corr']:.3f}")
    
    return metrics


def update_method_metrics(method: str, results_dir: Path, new_geometry: Dict[str, float]):
    """Update the final_metrics.json file with new geometry metrics."""
    
    metrics_file = results_dir / method / "final_metrics.json"
    
    if not metrics_file.exists():
        print(f"    ✗ Metrics file not found: {metrics_file}")
        return
    
    try:
        # Load existing metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Update geometry metrics in eval split
        if 'latent' in metrics and 'eval' in metrics['latent']:
            if 'geometry' not in metrics['latent']['eval']:
                metrics['latent']['eval']['geometry'] = {}
            
            metrics['latent']['eval']['geometry'].update(new_geometry)
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"    ✓ Updated geometry metrics in {metrics_file}")
        else:
            print(f"    ✗ Unexpected metrics structure in {metrics_file}")
            
    except Exception as e:
        print(f"    ✗ Error updating metrics file: {e}")


def main():
    """Main function to recompute geometry metrics using PSD reference."""
    
    # Configuration
    results_dir = Path("/rds/general/user/lrh24/home/thesis/code/Results")
    data_path = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean"
    split = "eval"  # Focus on evaluation split
    max_psd_samples = None  # Set to e.g. 1000 for testing, None for full dataset
    max_geom_samples = 5000  # Max samples for geometry computation (for efficiency)
    
    # All available methods (ordered by method groups)
    small_aggregated = ["tuh-ctm_cma_avg", "tuh-ctm_nn_avg", "tuh-hopf_avg", "tuh-jr_avg", "tuh-wong_wang_avg", "tuh-pca_avg", "tuh-psd_ae_avg"]
    medium_unrestricted = ["tuh-ctm_nn_pc", "tuh-hopf_pc", "tuh-jr_pc", "tuh-c22", "tuh-pca_pc", "tuh-psd_ae_pc", "tuh-eegnet"]
    all_methods = small_aggregated + medium_unrestricted
    
    print("=" * 80)
    print("RECOMPUTING GEOMETRY METRICS WITH PSD REFERENCE")
    print("=" * 80)
    
    # Step 1: Load and compute PSD features (reference space)
    print(f"\n1. Loading EEG data and computing PSD features...")
    
    # Separate methods by group: small_aggregated uses averaged PSD, medium_unrestricted uses per-channel PSD
    avg_psd_methods = small_aggregated  # All small_aggregated methods use averaged PSD
    pc_psd_methods = medium_unrestricted  # All medium_unrestricted methods use per-channel PSD
    
    try:
        # Compute averaged PSD features for small_aggregated group
        print(f"   Computing averaged PSD features for small_aggregated: {[m.replace('tuh-', '') for m in avg_psd_methods]}")
        psd_features_avg, psd_sample_ids_avg = load_and_compute_psd_features(
            data_path, split, max_samples=max_psd_samples, use_average=True
        )
        print(f"   ✓ Averaged PSD features: {psd_features_avg.shape}")
        
        # Compute per-channel PSD features for medium_unrestricted group  
        print(f"   Computing per-channel PSD features for medium_unrestricted: {[m.replace('tuh-', '') for m in pc_psd_methods]}")
        psd_features_pc, psd_sample_ids_pc = load_and_compute_psd_features(
            data_path, split, max_samples=max_psd_samples, use_average=False
        )
        print(f"   ✓ Per-channel PSD features: {psd_features_pc.shape}")
        
        # Verify same sample IDs in both
        if psd_sample_ids_avg != psd_sample_ids_pc:
            print(f"   ⚠️  Sample ID mismatch between averaged and per-channel PSDs")
            
    except Exception as e:
        print(f"   ✗ Failed to compute PSD features: {e}")
        return
    
    # Step 2: Process each method
    print(f"\n2. Processing methods...")
    updated_methods = []
    
    for method in all_methods:
        print(f"\n  Processing {method}...")
        
        # Check if method directory exists
        method_dir = results_dir / method
        if not method_dir.exists():
            print(f"    ✗ Method directory not found: {method_dir}")
            continue
        
        # Load latent features
        latent_result = load_latent_features(method, results_dir, split)
        if latent_result is None:
            continue
        
        latent_features, latent_sample_ids = latent_result
        
        try:
            # Choose appropriate PSD features based on method group
            if method in avg_psd_methods:  # small_aggregated group
                psd_features = psd_features_avg
                psd_sample_ids = psd_sample_ids_avg
                psd_type = "averaged (small_aggregated)"
            elif method in pc_psd_methods:  # medium_unrestricted group
                psd_features = psd_features_pc
                psd_sample_ids = psd_sample_ids_pc
                psd_type = "per-channel (medium_unrestricted)"
            else:
                print(f"    ✗ Unknown method group for {method}, skipping")
                continue
            
            print(f"    Using {psd_type} PSD reference ({psd_features.shape[1]} features)")
            
            # Align features by sample ID
            aligned_latent, aligned_psd, common_ids = align_features_by_sample_id(
                latent_features, latent_sample_ids, psd_features, psd_sample_ids
            )
            
            # Skip if too few samples
            if len(common_ids) < 50:
                print(f"    ✗ Too few aligned samples ({len(common_ids)}), skipping")
                continue
            
            # Compute new geometry metrics
            new_geometry = compute_geometry_metrics(
                aligned_latent, aligned_psd, max_samples=max_geom_samples
            )
            
            # Update the metrics file
            update_method_metrics(method, results_dir, new_geometry)
            updated_methods.append(method)
            
        except Exception as e:
            print(f"    ✗ Error processing {method}: {e}")
            continue
    
    # Step 3: Summary
    print(f"\n3. Summary...")
    print(f"   ✓ Successfully updated {len(updated_methods)}/{len(all_methods)} methods")
    print(f"   ✓ Updated methods: {[m.replace('tuh-', '') for m in updated_methods]}")
    
    if updated_methods:
        print(f"\n   The geometry metrics now compare:")
        print(f"   - High-D space: EEG PSD features")
        print(f"     • Small aggregated: {len(avg_psd_methods)} methods using averaged PSD ({psd_features_avg.shape[1]} dims)")
        print(f"     • Medium unrestricted: {len(pc_psd_methods)} methods using per-channel PSD ({psd_features_pc.shape[1]} dims)")
        print(f"   - Low-D space: Method latent features (various dimensions)")
        print(f"   - Subsampling: Up to {max_geom_samples} samples for efficiency")
        print(f"   - Metrics: trustworthiness, continuity, distance correlation")
        
        print(f"\n   To see the updated results, re-run metrics_and_plots.py")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
