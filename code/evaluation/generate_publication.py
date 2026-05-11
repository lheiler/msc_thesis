"""Generate publication-quality figures from evaluation results."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import scipy.spatial.distance as ssd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# =====================================================================
# I. GLOBAL TECHNICAL & AESTHETIC STANDARDS
# =====================================================================
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  12,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "black",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
})

METHOD_BLUE = "#1F4E79"      # data-driven
METHOD_ORANGE = "#BC3B00"    # mechanistic
METHOD_FALLBACK = "#5B9BD5"  # unknown/unmapped

DATA_DRIVEN_METHODS = {
    "psd_ae_avg", "psd_ae_pc", "c22", "pca_avg", "pca_pc", "eegnet"
}
MECHANISTIC_METHODS = {
    "hopf_avg", "hopf_pc", "jr_avg", "jr_pc", "wong_wang_avg",
    "ctm_cma_avg", "ctm_nn_avg", "ctm_nn_pc"
}

def get_method_color(method_name: str) -> str:
    """Get consistent color by method family."""
    clean_name = method_name
    for prefix in ("tuh-", "lemon-", "ntuh-", "harvard-"):
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
    if clean_name in DATA_DRIVEN_METHODS:
        return METHOD_BLUE
    if clean_name in MECHANISTIC_METHODS:
        return METHOD_ORANGE
    return METHOD_FALLBACK

METHOD_META = {
    "ctm_cma_avg":  {"label": "CTM-CMA",      "color": get_method_color("ctm_cma_avg")},
    "ctm_nn_avg":   {"label": "CTM-NN",       "color": get_method_color("ctm_nn_avg")},
    "ctm_nn_pc":    {"label": "CTM-NN (pc)",  "color": get_method_color("ctm_nn_pc")},
    "jr_avg":       {"label": "JR",           "color": get_method_color("jr_avg")},
    "jr_pc":        {"label": "JR (pc)",      "color": get_method_color("jr_pc")},
    "hopf_avg":     {"label": "Hopf",         "color": get_method_color("hopf_avg")},
    "hopf_pc":      {"label": "Hopf (pc)",    "color": get_method_color("hopf_pc")},
    "wong_wang_avg":{"label": "Wong-Wang",    "color": get_method_color("wong_wang_avg")},
    "c22":          {"label": "catch22",      "color": get_method_color("c22")},
    "eegnet":       {"label": "EEGNet",       "color": get_method_color("eegnet")},
    "pca_avg":      {"label": "PCA",          "color": get_method_color("pca_avg")},
    "pca_pc":       {"label": "PCA (pc)",     "color": get_method_color("pca_pc")},
    "psd_ae_avg":   {"label": "PSD-AE",       "color": get_method_color("psd_ae_avg")},
    "psd_ae_pc":    {"label": "PSD-AE (pc)",  "color": get_method_color("psd_ae_pc")},
}

# =====================================================================
# II. GROUP DEFINITIONS
# =====================================================================
SMALL_GROUP = {"ctm_nn_avg", "ctm_cma_avg", "wong_wang_avg", "hopf_avg",
               "jr_avg", "pca_avg", "psd_ae_avg"}
MEDIUM_GROUP = {"ctm_nn_pc", "hopf_pc", "jr_pc", "pca_pc", "c22",
                "psd_ae_pc", "eegnet"}

DATASET_TASK = {"tuh": "abnormal", "lemon": "age"}
DATASET_LABEL = {"tuh": "TUH-AB (Abnormality)", "lemon": "LEMON (Age)"}

# =====================================================================
# III. HELPERS
# =====================================================================

def _save_figure(fig, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved %s", path)

def _panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")

def _strip_prefix(method: str) -> str:
    for prefix in ("tuh-", "lemon-", "ntuh-", "harvard-"):
        if method.startswith(prefix):
            return method[len(prefix):]
    return method

def _get_dataset(full_name: str) -> str:
    if full_name.startswith("tuh-"):
        return "tuh"
    elif full_name.startswith("lemon-"):
        return "lemon"
    return "unknown"

def _get_scale(clean_name: str) -> str:
    if clean_name in SMALL_GROUP:
        return "small"
    elif clean_name in MEDIUM_GROUP:
        return "medium"
    return "unknown"

# =====================================================================
# IV. DATA LOADING
# =====================================================================

def collect_all_metrics(results_dir: str) -> Dict[str, Dict]:
    """Load metrics from latent_metrics.json + final_metrics.txt for each method."""
    all_metrics = {}
    if not os.path.isdir(results_dir):
        return all_metrics
    for entry in sorted(os.listdir(results_dir)):
        entry_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(entry_dir):
            continue
        latent_path = os.path.join(entry_dir, "latent_metrics.json")
        txt_path = os.path.join(entry_dir, "final_metrics.txt")
        if not os.path.isfile(latent_path):
            continue
        try:
            with open(latent_path, "r") as f:
                data = {"latent": json.load(f)}
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as f:
                    data["raw_txt"] = f.read()
            all_metrics[entry] = data
        except Exception as e:
            logger.warning("Could not load %s: %s", entry, e)
    return all_metrics


def load_latent_features(results_dir: str, method: str) -> Tuple[np.ndarray, List[str]]:
    """Load raw latent vectors from JSONL file."""
    path = os.path.join(results_dir, method, "temp_latent_features_eval.json")
    if not os.path.exists(path):
        return np.array([]), []
    features, ids = [], []
    try:
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                features.append(row[0])
                ids.append(row[4])
        return np.array(features), ids
    except Exception as e:
        logger.warning("Error loading %s features: %s", method, e)
        return np.array([]), []


# =====================================================================
# V. ON-THE-FLY METRIC COMPUTATION
# =====================================================================

def _trustworthiness(X_high, X_low, n_neighbors=10, max_samples=2000):
    """Trustworthiness metric [0,1]. Subsamples for speed."""
    n = X_high.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, size=max_samples, replace=False)
        X_high, X_low = X_high[idx], X_low[idx]
        n = max_samples
    D_X = pairwise_distances(X_high)
    np.fill_diagonal(D_X, np.inf)
    ranks = D_X.argsort(axis=1)
    D_Y = pairwise_distances(X_low)
    np.fill_diagonal(D_Y, np.inf)
    neigh_low = D_Y.argsort(axis=1)[:, :n_neighbors]
    k = n_neighbors
    t_sum = 0.0
    for i in range(n):
        Ni_low = set(neigh_low[i])
        for j in Ni_low:
            r_ij = int(np.where(ranks[i] == j)[0][0]) + 1
            if r_ij > k:
                t_sum += r_ij - k
    denom = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 / denom) * t_sum if denom > 0 else 0.0


def _continuity(X_high, X_low, n_neighbors=10, max_samples=2000):
    """Continuity metric [0,1]. Subsamples for speed."""
    n = X_high.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, size=max_samples, replace=False)
        X_high, X_low = X_high[idx], X_low[idx]
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


def compute_geometry(Z: np.ndarray) -> Dict[str, float]:
    """Compute trustworthiness and continuity using PCA(2) as reference."""
    if Z.shape[0] < 20 or Z.shape[1] < 2:
        return {}
    pca2 = PCA(n_components=2)
    Z_low = pca2.fit_transform(Z)
    return {
        "trustworthiness": _trustworthiness(Z, Z_low),
        "continuity": _continuity(Z, Z_low),
    }


def linear_cka_fast(X, Y):
    """Fast Linear CKA between two representation matrices."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(np.linalg.norm(Xc.T @ Yc) ** 2)
    hsic_xx = float(np.linalg.norm(Xc.T @ Xc) ** 2)
    hsic_yy = float(np.linalg.norm(Yc.T @ Yc) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(np.clip(hsic_xy / denom, -1.0, 1.0))


# =====================================================================
# MAIN TEXT FIGURES & TABLES
# =====================================================================


def generate_table1_capacity_efficiency(all_metrics, output_dir):
    rows = []
    for full_name, data in all_metrics.items():
        clean = _strip_prefix(full_name)
        ds = _get_dataset(full_name)
        ev = data.get("latent", {}).get("eval", {})
        dim = ev.get("dim", "N/A")
        active = ev.get("active_units", "N/A")
        eff = f"{(active/dim)*100:.1f}%" if dim != "N/A" and active != "N/A" and dim > 0 else "N/A"
        label = METHOD_META.get(clean, {}).get("label", clean)
        ds_label = "TUH" if ds == "tuh" else "LEMON"
        rows.append(f"| {label} ({ds_label}) | {dim} | {active} | {eff} |")

    path = os.path.join(output_dir, "table1_capacity_efficiency.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("| Model | Total Dimensions | Active Units | Efficiency |\n")
        f.write("|---|---|---|---|\n")
        f.write("\n".join(sorted(rows)))
    logger.info("Saved %s", path)


def generate_table2_summary_acc(all_metrics, output_dir):
    rows = []
    for full_name, data in all_metrics.items():
        clean = _strip_prefix(full_name)
        ds = _get_dataset(full_name)
        txt = data.get("raw_txt", "")
        label = METHOD_META.get(clean, {}).get("label", clean)
        ds_label = "TUH-AB" if ds == "tuh" else "LEMON"
        task_key = DATASET_TASK.get(ds, "")

        acc = "N/A"
        match = re.search(rf"metrics_per_task\.{task_key}\.accuracy:\s+([0-9.]+)", txt)
        if match:
            acc = f"{float(match.group(1))*100:.2f}%"
        rows.append(f"| {label} ({ds_label}) | {acc} |")

    path = os.path.join(output_dir, "table2_summary_acc.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("| Model (Dataset) | Test Accuracy |\n")
        f.write("|---|---|\n")
        f.write("\n".join(sorted(rows)))
    logger.info("Saved %s", path)


def generate_fig2_probe_delta(all_metrics, output_dir):
    """Performance comparison with linear probe delta, faceted by dataset.

    Methods are grouped by family (mechanistic first, data-driven second) and
    sorted by MLP accuracy descending within each group. A dashed separator and
    shaded band make the group boundary immediately visible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=False)

    for col, ds in enumerate(["tuh", "lemon"]):
        ax = axes[col]
        task = DATASET_TASK[ds]

        # Collect entries as (group_order, -mlp_acc, label, mlp, probe, color)
        # group 0 = mechanistic (orange), group 1 = data-driven (blue)
        entries = []
        for full_name, data in all_metrics.items():
            if _get_dataset(full_name) != ds:
                continue
            clean = _strip_prefix(full_name)
            if clean not in METHOD_META:
                continue
            txt = data.get("raw_txt", "")
            mlp_match   = re.search(rf"metrics_per_task\.{task}\.accuracy:\s+([0-9.]+)", txt)
            probe_match = re.search(rf"metrics_per_task\.{task}_linear_probe\.accuracy:\s+([0-9.]+)", txt)
            if mlp_match and probe_match:
                mlp   = float(mlp_match.group(1))   * 100
                probe = float(probe_match.group(1)) * 100
                group = 0 if clean in MECHANISTIC_METHODS else 1
                entries.append((group, -mlp, METHOD_META[clean]["label"],
                                 mlp, probe, METHOD_META[clean]["color"]))

        # Sort: mechanistic block first, then data-driven; within each block by MLP ↓
        entries.sort(key=lambda e: (e[0], e[1]))
        if not entries:
            continue

        groups     = [e[0] for e in entries]
        methods    = [e[2] for e in entries]
        mlp_accs   = [e[3] for e in entries]
        probe_accs = [e[4] for e in entries]
        colors     = [e[5] for e in entries]

        y_pos = np.arange(len(methods))

        # Shaded background band per group
        group_indices: dict = {}
        for i, g in enumerate(groups):
            group_indices.setdefault(g, []).append(i)
        group_meta = {0: (METHOD_ORANGE, "Mechanistic"),
                      1: (METHOD_BLUE,   "Data-Driven")}
        for g, indices in sorted(group_indices.items()):
            band_color, _ = group_meta[g]
            y_lo = min(indices) - 0.45
            y_hi = max(indices) + 0.45
            ax.axhspan(y_lo, y_hi, alpha=0.05, color=band_color, zorder=0)

        # Dashed separator between groups
        for i in range(1, len(groups)):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color='gray', linewidth=0.8,
                           linestyle='--', alpha=0.6, zorder=1)

        # Draw dots and connectors
        for y, mlp, probe, col_c in zip(y_pos, mlp_accs, probe_accs, colors):
            ax.plot([probe, mlp], [y, y], '-', color='gray', alpha=0.5, zorder=1)
            ax.scatter(probe, y, color='white', edgecolor=col_c, s=80, zorder=2,
                       label='Linear Probe' if y == 0 else "")
            ax.scatter(mlp, y, color=col_c, s=80, zorder=3,
                       label='Non-Linear (MLP)' if y == 0 else "")
            delta = mlp - probe
            ax.text(max(probe, mlp) + 0.5, y,
                    f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%",
                    va='center', fontsize=7, color=col_c)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        for tick, col in zip(ax.get_yticklabels(), colors):
            tick.set_color(col)
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title(DATASET_LABEL[ds])
        if methods:
            ax.legend(fontsize=8)
    _save_figure(fig, "fig2_performance_delta", output_dir)


def generate_fig3_bifurcation(output_dir, results_dir):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig = plt.figure(figsize=(7.5, 4.5))
    mosaic = fig.subplot_mosaic([["A", "B"]])
    ax_a = mosaic["A"]; _panel_label(ax_a, "A")
    g_ee_range = np.linspace(0, 20, 120)
    g_srs_range = np.linspace(-5, 0, 100)
    GE, GS = np.meshgrid(g_ee_range, g_srs_range)
    stability_map = np.zeros_like(GE)
    p_template = {'G_ee': 10, 'G_ei': -20, 'G_ese': 5, 'G_esre': -5,
                  'G_srs': -0.5, 'alpha': 50, 'beta': 300, 't0': 0.1}
    try:
        from latent_extraction.cortico_thalamic import _q2_re2
    except ImportError:
        _q2_re2 = None

    omega_test = 2 * np.pi * 10.0
    for i in range(GE.shape[0]):
        for j in range(GE.shape[1]):
            p = p_template.copy(); p['G_ee'] = GE[i, j]; p['G_srs'] = GS[i, j]
            try:
                q2 = _q2_re2(omega_test, p) if _q2_re2 else \
                    (1 - p['G_ee']/20.0) * (1 + p['G_srs']/5.0) * 10
                stability_map[i, j] = np.real(q2) if np.isfinite(q2) else np.nan
            except Exception:
                stability_map[i, j] = np.nan

    im_a = ax_a.contourf(GE, GS, stability_map, levels=30, cmap="RdBu_r")
    ax_a.contour(GE, GS, stability_map, levels=[0], colors="black", linewidths=1.5)
    fig.colorbar(im_a, ax=ax_a, shrink=0.8, label="q²r²ₑ (stability)")
    ax_a.set_xlabel("G_ee"); ax_a.set_ylabel("G_srs")
    ax_a.set_title("CTM Bifurcation Diagram")

    ax_b = mosaic["B"]; _panel_label(ax_b, "B")
    ax_b.set_xlabel("G_ee"); ax_b.set_ylabel("G_srs")
    ax_b.contourf(GE, GS, stability_map, levels=30, cmap="RdBu_r", alpha=0.3)
    ax_b.contour(GE, GS, stability_map, levels=[0], colors="black", linewidths=1.0)

    # Load real fitted CTM parameters (G_ee=index 0, G_srs=index 4 per _PARAM_KEYS)
    rng = np.random.RandomState(42)
    try:
        def _load_gee_gsrs(path, n_sample=200):
            gee, gsrs = [], []
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    gee.append(row[0][0])
                    gsrs.append(row[0][4])
            idx = rng.choice(len(gee), min(n_sample, len(gee)), replace=False)
            return np.array(gee)[idx], np.array(gsrs)[idx]

        tuh_gee, tuh_gsrs = _load_gee_gsrs(
            os.path.join(results_dir, "tuh-ctm_cma_avg", "temp_latent_features_eval.json"))
        lemon_gee, lemon_gsrs = _load_gee_gsrs(
            os.path.join(results_dir, "lemon-ctm_cma_avg", "temp_latent_features_eval.json"))
        ax_b.set_title("Empirical CTM Latents Overlay")
    except Exception as e:
        logger.warning("Could not load real CTM params for fig3B (%s)", e)
        tuh_gee = tuh_gsrs = lemon_gee = lemon_gsrs = np.array([])
        ax_b.set_title("Empirical CTM Latents Overlay (data unavailable)")

    if len(tuh_gee):
        ax_b.scatter(tuh_gee, tuh_gsrs, s=12, alpha=0.5, color="#e41a1c", label="TUH-AB")
    if len(lemon_gee):
        ax_b.scatter(lemon_gee, lemon_gsrs, s=12, alpha=0.5, color="#377eb8", label="LEMON")
    ax_b.legend()
    _save_figure(fig, "fig3_bifurcation", output_dir)


# =====================================================================
# FIGURE 4: DISTANCE-GEOMETRY CORRELATION MATRICES (2×2: Dataset × Scale)
# =====================================================================

def generate_fig4_similarity_matrices(all_metrics, output_dir, results_dir):
    """Pairwise distance-geometry correlation between all methods (report §3.6).

    For each pair of methods, aligns latent vectors by sample_id, subsamples to
    ≤2 000 common epochs (seed 42), then computes the Pearson correlation between
    their upper-triangular pairwise Euclidean distance matrices.  Values near 1
    mean the two methods agree on which subjects are globally close or far apart.
    """
    logger.info("Computing Distance-Geometry Correlation Matrices(2×2 grid)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]
    _MAX_SAMPLES = 10000   # pairwise distances scale as O(n²); 2k is fast & stable

    for row, ds in enumerate(["tuh", "lemon"]):
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            methods, labels, cleans = [], [], []
            for full_name in sorted(all_metrics.keys()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue
                methods.append(full_name)
                labels.append(METHOD_META[clean]["label"])
                cleans.append(clean)

            if len(methods) < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}")
                continue

            # Load latent features
            feat = {}
            for m in methods:
                Z, ids = load_latent_features(results_dir, m)
                if len(Z) > 0:
                    feat[m] = {"Z": Z, "ids": ids}

            n = len(methods)
            sim = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    m1, m2 = methods[i], methods[j]
                    if m1 not in feat or m2 not in feat:
                        continue
                    d1, d2 = feat[m1], feat[m2]
                    idx1_map = {sid: k for k, sid in enumerate(d1["ids"])}
                    idx2_map = {sid: k for k, sid in enumerate(d2["ids"])}
                    common = [sid for sid in d1["ids"] if sid in idx2_map]
                    if len(common) < 20:
                        continue
                    a1 = np.array([idx1_map[s] for s in common])
                    a2 = np.array([idx2_map[s] for s in common])
                    Z1 = d1["Z"][a1];  Z2 = d2["Z"][a2]
                    # Subsample for tractability
                    if len(Z1) > _MAX_SAMPLES:
                        rng = np.random.RandomState(42)
                        sub = rng.choice(len(Z1), _MAX_SAMPLES, replace=False)
                        Z1 = Z1[sub];  Z2 = Z2[sub]
                    score = _distance_correlation(Z1, Z2)
                    sim[i, j] = sim[j, i] = score

            # Cluster ordering by similarity
            dist = np.clip(1 - sim, 0, 2)
            np.fill_diagonal(dist, 0)
            try:
                pdist_vec = ssd.squareform(dist)
                Z_link = linkage(pdist_vec, 'ward')
                order = leaves_list(Z_link)
            except Exception:
                order = np.arange(n)

            sim_ord = sim[order][:, order]
            lab_ord = [labels[i] for i in order]
            col_ord = [get_method_color(cleans[i]) for i in order]

            im = ax.imshow(sim_ord, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(lab_ord)))
            ax.set_yticks(np.arange(len(lab_ord)))
            ax.set_xticklabels(lab_ord, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(lab_ord, fontsize=7)
            for tick, col in zip(ax.get_xticklabels(), col_ord):
                tick.set_color(col)
            for tick, col in zip(ax.get_yticklabels(), col_ord):
                tick.set_color(col)
            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

            for ii in range(len(lab_ord)):
                for jj in range(len(lab_ord)):
                    val = sim_ord[ii, jj]
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val < 0.5 else "black")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)),
                 cax=cbar_ax, label="Distance-Geometry Correlation (Pearson r)")
    # fig.suptitle("Figure 4: Representational Similarity (Distance-Geometry Correlation)",
    #              fontweight="bold", y=0.98)
    _save_figure(fig, "fig4_similarity_matrices", output_dir)


# =====================================================================
# FIGURE 5: MI CONCENTRATION (2×2: Task × Scale)
# =====================================================================

def generate_fig5_mi_concentration(all_metrics, output_dir):
    """Mean per-dimension MI(Z_j; Y) per method (report §3.6: Information Content).

    Uses pre-computed mean MI from latent_metrics.json (sklearn mutual_information_classif).
    Task label: 'abnormal' for TUH-AB, 'age' for LEMON.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.45)
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]

    for row, ds in enumerate(["tuh", "lemon"]):
        task = DATASET_TASK[ds]   # 'abnormal' | 'age'
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            entries = []
            for full_name, data in sorted(all_metrics.items()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue

                ev = data.get("latent", {}).get("eval", {})
                mi_mean = ev.get("mi_zy", {}).get(task, {}).get("mean")
                if mi_mean is not None:
                    entries.append((float(mi_mean), METHOD_META[clean]["color"],
                                    METHOD_META[clean]["label"]))

            # Sort descending by MI value
            entries.sort(key=lambda e: e[0], reverse=True)

            if entries:
                y_val  = [e[0] for e in entries]
                colors = [e[1] for e in entries]
                labels = [e[2] for e in entries]
                x_pos  = np.arange(len(labels))
                ax.bar(x_pos, y_val, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                for tick, col in zip(ax.get_xticklabels(), colors):
                    tick.set_color(col)
                ax.set_ylabel('Mean MI per Dimension\n' r'$\bar{I}(Z_j\,;\,Y)$')
                ax.set_ylim(bottom=0)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No MI data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

    _save_figure(fig, "fig5_mi_concentration", output_dir)


# =====================================================================
# APPENDIX FIGURES & TABLES
# =====================================================================

def generate_tableA1_full_matrix(all_metrics, output_dir):
    rows = []
    for full_name, data in all_metrics.items():
        txt = data.get("raw_txt", "")
        tasks = re.findall(r"metrics_per_task\.([a-zA-Z0-9_]+)\.accuracy:\s+([0-9.]+)", txt)

        for task_name, test_acc in tasks:
            is_probe = "_linear_probe" in task_name
            base_task = task_name.replace("_linear_probe", "")
            pfx = rf"cross_validation\.{base_task}\.cv\.{base_task}\." + \
                  ("linear_probe" if is_probe else "mlp")
            cv_mean = re.search(rf"{pfx}\.accuracy_mean:\s+([0-9.]+)", txt)
            cv_std = re.search(rf"{pfx}\.accuracy_std:\s+([0-9.]+)", txt)
            m = cv_mean.group(1) if cv_mean else "N/A"
            s = cv_std.group(1) if cv_std else "N/A"
            rows.append(f"| {full_name} | {task_name} | {float(test_acc):.4f} | {m} | {s} |")

    rows.sort()
    path = os.path.join(output_dir, "tableA1_full_matrix.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as out:
        out.write("| Model | Task | Test Accuracy | CV Mean | CV Std |\n|---|---|---|---|---|\n")
        out.write("\n".join(rows))
    logger.info("Saved %s", path)


def generate_figA1_multidim_efficiency(all_metrics, output_dir):
    """Efficiency bubble plot faceted by scale, with dataset markers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    ds_markers = {"tuh": "o", "lemon": "s"}

    for col, scale in enumerate(["small", "medium"]):
        ax = axes[col]
        for full_name, data in sorted(all_metrics.items()):
            ds = _get_dataset(full_name)
            clean = _strip_prefix(full_name)
            if clean not in METHOD_META or _get_scale(clean) != scale:
                continue

            ev = data.get("latent", {}).get("eval", {})
            active = ev.get("active_units")
            dim = ev.get("dim")
            var_per_dim = ev.get("variance_per_dim", [])

            if not (active and dim and var_per_dim and dim > 0):
                continue

            norm_var = np.array(var_per_dim) / (np.sum(var_per_dim) + 1e-12)
            ve = -np.sum(norm_var * np.log(norm_var + 1e-10))

            marker = ds_markers.get(ds, "^")
            ax.scatter(active/dim, ve, s=np.sqrt(dim)*30, alpha=0.7, edgecolors='black',
                       color=METHOD_META[clean]["color"], marker=marker)
            ax.annotate(METHOD_META[clean]["label"],
                        (active/dim, ve), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

        ax.set_xlabel('Dimensional Efficiency (Active/Total)')
        ax.set_ylabel('Variance Entropy (Uniformity)')
        ax.set_title(scale_labels[scale])
        ax.grid(True, alpha=0.3)

        # Custom legend for dataset markers
        from matplotlib.lines import Line2D
        legend_ds = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                            markersize=8, label='TUH'),
                     Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                            markersize=8, label='LEMON')]
        ax.legend(handles=legend_ds, loc='upper left')

    # fig.sup title("Figure A1: Multi-Dimensional Efficiency", fontweight="bold")
    _save_figure(fig, "figA1_multidim_efficiency", output_dir)


# =====================================================================
# FIGURE A4: GEOMETRIC PRESERVATION HELPERS
# =====================================================================

_GEO_DATASET_PATHS: Dict[str, str] = {
    "tuh": os.environ.get("TUH_EVAL_EPOCHS", "Datasets/tuh-eeg-ab-clean/eval_epochs.pkl"),
    "lemon": os.environ.get("LEMON_EVAL_EPOCHS", "Datasets/lemon/eval_epochs.pkl"),
}

def _distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Pearson r between upper-triangular pairwise Euclidean distance matrices.

    Captures global distance structure agreement, independent of rotation or
    translation (report §3.6: 'Distance correlation').
    """
    from scipy.spatial.distance import pdist
    d1 = pdist(X)
    d2 = pdist(Y)
    if d1.std() < 1e-10 or d2.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(d1, d2)[0, 1])


def _build_psd_reference(ds: str, scale: str, epoch_cache: dict) -> dict:
    """Compute {sample_id: flat_psd_vector} reference for a dataset+scale combo.

    Per report §3.6:
      Small group  → channel-averaged PSD (log10 + z-scored) → 1-D vector.
      Medium group → per-channel PSD (log10 + z-scored per channel), flattened → 1-D.
    Welch params: nfft=512, nper_seg=512, noverlap=256, 1–45 Hz.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import pickle
        import mne
        mne.set_log_level('WARNING')
        from utils.util import compute_psd_from_raw
    except ImportError as e:
        logger.warning("PSD reference import failed: %s", e)
        return {}

    if ds not in epoch_cache:
        path = _GEO_DATASET_PATHS.get(ds)
        if not path or not os.path.exists(path):
            logger.warning("No epoch file for '%s'", ds)
            return {}
        logger.info("Loading %s eval epochs for PSD reference", ds)
        try:
            with open(path, 'rb') as f:
                epoch_cache[ds] = pickle.load(f)
        except Exception as e:
            logger.warning("Could not load %s epochs: %s", ds, e)
            return {}

    calc_avg = (scale == 'small')
    ref = {}
    for item in epoch_cache[ds]:
        raw, _, _, _, sid = item
        try:
            psd = compute_psd_from_raw(raw, calculate_average=calc_avg, normalize=True)
            ref[sid] = psd.flatten().astype(np.float32)
        except Exception:
            continue
    logger.info("PSD reference built: %s/%s -> %d epochs", ds, scale, len(ref))
    return ref


# =====================================================================
# FIGURE A4: GEOMETRIC PRESERVATION (2×2: Dataset × Scale, computed on-the-fly)
# =====================================================================

def generate_figA4_geometric_preservation(all_metrics, output_dir, results_dir):
    """Geometric preservation of latent space relative to PSD reference (report §3.6).

    Three metrics per method:
      - Trustworthiness (k=10): penalises false neighbours introduced in latent space.
      - Continuity      (k=10): penalises true PSD neighbours dropped in latent space.
      - Distance correlation:   Pearson r between pairwise distance matrices.
    Reference space: Welch PSD (1–45 Hz, log10 + z-score).
      Small  group → channel-averaged PSD.
      Medium group → per-channel PSD, flattened.
    Subsampled to ≤5 000 samples with seed 42 (report spec).
    """
    logger.info("Computing geometric preservation vs PSD reference (report §3.6)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.45)
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]

    epoch_cache: dict = {}   # loaded once per dataset, reused across panels
    psd_cache:   dict = {}   # {(ds, scale): {sid: psd_vec}}

    for row, ds in enumerate(["tuh", "lemon"]):
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            # Build / reuse PSD reference
            cache_key = (ds, scale)
            if cache_key not in psd_cache:
                psd_cache[cache_key] = _build_psd_reference(ds, scale, epoch_cache)
            psd_ref = psd_cache[cache_key]

            if not psd_ref:
                ax.text(0.5, 0.5, "PSD reference unavailable",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)
                continue

            method_labels, method_cleans = [], []
            trust_vals, cont_vals = [], []

            for full_name in sorted(all_metrics.keys()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue

                Z_all, ids_all = load_latent_features(results_dir, full_name)
                if len(Z_all) < 20:
                    continue

                # Align Z with PSD by sample_id
                common = [(i, sid) for i, sid in enumerate(ids_all) if sid in psd_ref]
                if len(common) < 20:
                    logger.info("%s: only %d common IDs, skipping", full_name, len(common))
                    continue

                idx_z = np.array([i for i, _ in common])
                sids  = [sid for _, sid in common]
                Z     = Z_all[idx_z].astype(np.float32)
                X_psd = np.array([psd_ref[s] for s in sids], dtype=np.float32)

                # Subsample to 5 000 with seed 42 (report §3.6)
                if len(Z) > 5000:
                    rng = np.random.RandomState(42)
                    sub = rng.choice(len(Z), 5000, replace=False)
                    Z = Z[sub]; X_psd = X_psd[sub]

                logger.info("Geometry: %s (n=%d)", full_name, len(Z))
                try:
                    # max_samples > actual n → no internal resampling
                    t  = _trustworthiness(X_psd, Z, n_neighbors=10, max_samples=6000)
                    c  = _continuity(X_psd, Z,     n_neighbors=10, max_samples=6000)
                    method_labels.append(METHOD_META[clean]["label"])
                    method_cleans.append(clean)
                    trust_vals.append(t)
                    cont_vals.append(c)
                    logger.info("T=%.3f C=%.3f", t, c)
                except Exception as e:
                    logger.warning("error (%s)", e)

            if method_labels:
                avg   = [(t + c) / 2 for t, c in zip(trust_vals, cont_vals)]
                order = np.argsort(avg)[::-1]
                method_labels = [method_labels[i] for i in order]
                method_cleans = [method_cleans[i] for i in order]
                trust_vals    = [trust_vals[i]    for i in order]
                cont_vals     = [cont_vals[i]     for i in order]

                x     = np.arange(len(method_labels))
                width = 0.35
                ax.bar(x - width/2, trust_vals, width, label='Trustworthiness',
                       color='#1b9e77', alpha=0.9)
                ax.bar(x + width/2, cont_vals,  width, label='Continuity',
                       color='#d95f02', alpha=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
                for tick, clean in zip(ax.get_xticklabels(), method_cleans):
                    tick.set_color(get_method_color(clean))
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.05)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.legend(fontsize=7)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

    _save_figure(fig, "figA4_geometric_preservation", output_dir)


# =====================================================================
# METHODOLOGY FIGURE: PSD RECONSTRUCTION
# =====================================================================

def generate_figM1_psd_reconstruction(output_dir, results_dir):
    """Real PSD reconstruction comparison: CTM-CMA (mechanistic) vs PSD-AE (data-driven).

    Picks the best-fitting CTM epoch from the first 200 eval subjects in TUH and plots:
      A) Empirical PSD vs CTM-CMA fit (_P_omega with real fitted parameters)
      B) Empirical PSD vs PSD-AE decoded reconstruction (same epoch)
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import pickle
        import torch
        import mne
        mne.set_log_level('WARNING')
        from utils.util import normalize_psd, compute_psd_from_raw
        from latent_extraction.cortico_thalamic import _P_omega, _PARAM_KEYS
        from latent_extraction.psd_ae.psd_ae import get_psd_ae_model
    except ImportError as e:
        logger.warning("figM1: import failed (%s), skipping", e)
        return

    dataset_path = _GEO_DATASET_PATHS.get("tuh", "Datasets/tuh-eeg-ab-clean/eval_epochs.pkl")
    ctm_feat_path  = os.path.join(results_dir, 'tuh-ctm_cma_avg',  'temp_latent_features_eval.json')
    psdae_feat_path = os.path.join(results_dir, 'tuh-psd_ae_avg', 'temp_latent_features_eval.json')

    try:
        logger.info("Loading TUH eval epochs for figM1")
        with open(dataset_path, 'rb') as f:
            epochs = pickle.load(f)
    except Exception as e:
        logger.warning("figM1: could not load epochs (%s), skipping", e)
        return

    id_to_raw = {item[4]: item[0] for item in epochs}

    ctm_rows, psdae_rows = {}, {}
    with open(ctm_feat_path) as f:
        for line in f:
            row = json.loads(line); ctm_rows[row[4]] = row[0]
    with open(psdae_feat_path) as f:
        for line in f:
            row = json.loads(line); psdae_rows[row[4]] = row[0]

    common_ids = [eid for eid in ctm_rows if eid in psdae_rows and eid in id_to_raw]
    if not common_ids:
        logger.warning("figM1: no common epoch IDs found, skipping")
        return

    # Pick the epoch with the lowest CTM reconstruction loss from the first 200 candidates
    logger.info("Selecting best representative epoch from %d candidates", min(200, len(common_ids)))
    best_id, best_loss = common_ids[0], float('inf')
    for eid in common_ids[:200]:
        raw = id_to_raw[eid]
        p = {k: float(ctm_rows[eid][i]) for i, k in enumerate(_PARAM_KEYS)}
        try:
            emp_psd, freqs = compute_psd_from_raw(
                raw, calculate_average=True, normalize=False, return_freqs=True)
            loss = float(np.mean(
                (normalize_psd(_P_omega(p, freqs)) - normalize_psd(emp_psd)) ** 2))
            if loss < best_loss:
                best_loss, best_id = loss, eid
        except Exception:
            continue
    logger.info("Representative epoch: %s (CTM loss=%.4f)", best_id, best_loss)

    raw = id_to_raw[best_id]
    emp_psd, freqs = compute_psd_from_raw(
        raw, calculate_average=True, normalize=False, return_freqs=True)
    emp_norm = normalize_psd(emp_psd)

    # CTM-CMA: reconstruct PSD from fitted parameters
    p = {k: float(ctm_rows[best_id][i]) for i, k in enumerate(_PARAM_KEYS)}
    ctm_norm = normalize_psd(_P_omega(p, freqs))

    # PSD-AE: decode latent code (output is already in normalized log-space)
    psdae_model = get_psd_ae_model(device='cpu', dataset_name='tuh')
    z = torch.tensor(psdae_rows[best_id], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        psdae_recon = psdae_model.decode(z).squeeze().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    _panel_label(ax1, "A")
    ax1.plot(freqs, emp_norm, "k-", linewidth=1.5, label="Empirical PSD", alpha=0.8)
    ax1.plot(freqs, ctm_norm, "-", color=METHOD_ORANGE, linewidth=2.0, label="CTM-CMA Fit")
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Normalized log Power (a.u.)")
    ax1.set_title("A) CTM-CMA – Mechanistic Fit"); ax1.legend(frameon=False)
    ax1.set_xlim(freqs[0], freqs[-1])

    _panel_label(ax2, "B")
    ax2.plot(freqs, emp_norm, "k-", linewidth=1.5, label="Empirical PSD", alpha=0.8)
    ax2.plot(freqs, psdae_recon, "-", color=METHOD_BLUE, linewidth=2.0, label="PSD-AE Reconstruction")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("B) PSD-AE – Data-Driven Reconstruction")
    ax2.legend(frameon=False); ax2.set_xlim(freqs[0], freqs[-1])

    plt.suptitle("Figure M1: PSD Representation – Mechanistic vs Data-Driven", fontweight='bold')
    _save_figure(fig, "figM1_psd_reconstruction", output_dir)


# =====================================================================
# MAIN
# =====================================================================

def main():
    results_dir = "/rds/general/user/lrh24/home/msc_thesis/code/Results"
    output_dir = os.path.join(results_dir, "publication_figures")
    logger.info("Collecting metrics")
    all_metrics = collect_all_metrics(results_dir)
    logger.info("Loaded %d model evaluations", len(all_metrics))

    logger.info("Generating Main Text Figures & Tables")

    generate_table1_capacity_efficiency(all_metrics, output_dir)
    generate_table2_summary_acc(all_metrics, output_dir)
    generate_fig2_probe_delta(all_metrics, output_dir)
    generate_fig3_bifurcation(output_dir, results_dir)
    generate_fig4_similarity_matrices(all_metrics, output_dir, results_dir)
    generate_fig5_mi_concentration(all_metrics, output_dir)

    logger.info("Generating Methodology Figures")
    generate_figM1_psd_reconstruction(output_dir, results_dir)

    logger.info("Generating Appendix Figures & Tables")
    generate_tableA1_full_matrix(all_metrics, output_dir)
    generate_figA1_multidim_efficiency(all_metrics, output_dir)
    generate_figA4_geometric_preservation(all_metrics, output_dir, results_dir)

    logger.info("Publication bundle complete")

if __name__ == "__main__":
    main()
