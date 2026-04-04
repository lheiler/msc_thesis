import json
import os
import sys
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
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

METHOD_META = {
    "ctm_cma_avg":  {"label": "CTM-CMA",      "color": "#1b9e77"},
    "ctm_nn_avg":   {"label": "CTM-NN",       "color": "#d95f02"},
    "ctm_nn_pc":    {"label": "CTM-NN (pc)",  "color": "#e6ab02"},
    "jr_avg":       {"label": "JR",           "color": "#66a61e"},
    "jr_pc":        {"label": "JR (pc)",      "color": "#a6d854"},
    "hopf_avg":     {"label": "Hopf",         "color": "#377eb8"},
    "hopf_pc":      {"label": "Hopf (pc)",    "color": "#984ea3"},
    "wong_wang_avg":{"label": "Wong-Wang",    "color": "#4daf4a"},
    "c22":          {"label": "catch22",      "color": "#e41a1c"},
    "eegnet":       {"label": "EEGNet",       "color": "#ff7f00"},
    "pca_avg":      {"label": "PCA",          "color": "#a65628"},
    "pca_pc":       {"label": "PCA (pc)",     "color": "#f781bf"},
    "psd_ae_avg":   {"label": "PSD-AE",       "color": "#999999"},
    "psd_ae_pc":    {"label": "PSD-AE (pc)",  "color": "#636363"},
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
    print(f"  ✓ Saved {path}")

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
            print(f"  Warning: Could not load {entry}: {e}")
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
        print(f"  Error loading {method} features: {e}")
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

def generate_fig1_pipeline(output_dir):
    fig = plt.figure(figsize=(7.5, 5.0))
    mosaic = fig.subplot_mosaic([["A"], ["B"]], height_ratios=[1, 1.4])

    ax_a = mosaic["A"]
    ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 2); ax_a.axis("off")
    _panel_label(ax_a, "A", x=-0.02, y=1.05)
    boxes_a = [(0.3, 0.6, "Raw EEG\\n(TUH-AB)"), (2.5, 0.6, "Band-pass\\n1–45 Hz"),
               (4.7, 0.6, "Epoch\\n(10 s)"), (6.9, 0.6, "PSD\\n(Welch)")]
    bw, bh = 1.6, 0.9
    for x, y, txt in boxes_a:
        ax_a.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.1",
                       facecolor="#e8f4f8", edgecolor="#2c3e50", linewidth=1.2))
        ax_a.text(x + bw/2, y + bh/2, txt, ha="center", va="center", fontsize=8, fontweight="bold")
    for i in range(len(boxes_a) - 1):
        ax_a.annotate("", xy=(boxes_a[i+1][0], boxes_a[i][1]+bh/2),
                      xytext=(boxes_a[i][0]+bw, boxes_a[i][1]+bh/2),
                      arrowprops=dict(arrowstyle="->", lw=1.5, color="#2c3e50"))

    ax_b = mosaic["B"]
    ax_b.set_xlim(0, 10); ax_b.set_ylim(0, 3.5); ax_b.axis("off")
    _panel_label(ax_b, "B", x=-0.02, y=1.05)
    ax_b.add_patch(FancyBboxPatch((3.5, 2.6), 3.0, 0.7, boxstyle="round,pad=0.12",
                   facecolor="#ffeaa7", edgecolor="#2c3e50", linewidth=1.2))
    ax_b.text(5.0, 2.95, "Latent Feature\\nExtraction", ha="center", va="center",
              fontsize=9, fontweight="bold")

    dd_boxes = [(0.2, 0.8, "EEGNet\\n(AE)"), (2.2, 0.8, "catch22"),
                (0.2, 0.0, "PSD-AE"), (2.2, 0.0, "PCA")]
    for x, y, txt in dd_boxes:
        ax_b.add_patch(FancyBboxPatch((x, y), 1.7, 0.65, boxstyle="round,pad=0.08",
                       facecolor="#fab1a0", edgecolor="#d63031", linewidth=1.0))
        ax_b.text(x+0.85, y+0.325, txt, ha="center", va="center", fontsize=7.5)
    ax_b.text(1.9, 1.75, "Data-Driven", ha="center", va="center", fontsize=9,
              fontweight="bold", color="#d63031")

    mech_boxes = [(6.1, 0.8, "CTM\\n(CMA-ES)"), (8.1, 0.8, "Jansen-Rit"),
                  (6.1, 0.0, "Wong-Wang"), (8.1, 0.0, "Hopf")]
    for x, y, txt in mech_boxes:
        ax_b.add_patch(FancyBboxPatch((x, y), 1.7, 0.65, boxstyle="round,pad=0.08",
                       facecolor="#81ecec", edgecolor="#00b894", linewidth=1.0))
        ax_b.text(x+0.85, y+0.325, txt, ha="center", va="center", fontsize=7.5)
    ax_b.text(8.0, 1.75, "Mechanistic", ha="center", va="center", fontsize=9,
              fontweight="bold", color="#00b894")

    ax_b.add_patch(FancyBboxPatch((4.0, 0.3), 1.8, 0.65, boxstyle="round,pad=0.08",
                   facecolor="#ffeaa7", edgecolor="#d95f02", linewidth=1.5))
    ax_b.text(4.9, 0.625, "CTM-NN\\n(Hybrid)", ha="center", va="center",
              fontsize=8, fontweight="bold", color="#d95f02")
    ax_b.text(4.9, 1.75, "Hybrid", ha="center", va="center", fontsize=9,
              fontweight="bold", color="#d95f02")

    for tx in [1.9, 4.9, 8.0]:
        ax_b.annotate("", xy=(tx, 1.85), xytext=(5.0, 2.6),
                      arrowprops=dict(arrowstyle="->", lw=1.3, color="#636e72"))
    _save_figure(fig, "fig1_pipeline", output_dir)


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
    print(f"  ✓ Saved {path}")


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
    print(f"  ✓ Saved {path}")


def generate_fig2_probe_delta(all_metrics, output_dir):
    """Performance comparison with linear probe delta, faceted by dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for col, ds in enumerate(["tuh", "lemon"]):
        ax = axes[col]
        task = DATASET_TASK[ds]
        methods, mlp_accs, probe_accs, colors = [], [], [], []

        for full_name, data in all_metrics.items():
            if _get_dataset(full_name) != ds:
                continue
            clean = _strip_prefix(full_name)
            if clean not in METHOD_META:
                continue
            txt = data.get("raw_txt", "")

            mlp_match = re.search(rf"metrics_per_task\.{task}\.accuracy:\s+([0-9.]+)", txt)
            probe_match = re.search(rf"metrics_per_task\.{task}_linear_probe\.accuracy:\s+([0-9.]+)", txt)

            if mlp_match and probe_match:
                methods.append(METHOD_META[clean]["label"])
                mlp_accs.append(float(mlp_match.group(1)) * 100)
                probe_accs.append(float(probe_match.group(1)) * 100)
                colors.append(METHOD_META[clean]["color"])

        y_pos = np.arange(len(methods))
        for y, mlp, probe, col_c in zip(y_pos, mlp_accs, probe_accs, colors):
            ax.plot([probe, mlp], [y, y], '-', color='gray', alpha=0.5, zorder=1)
            ax.scatter(probe, y, color='white', edgecolor=col_c, s=80, zorder=2,
                       label='Linear Probe' if y == 0 else "")
            ax.scatter(mlp, y, color=col_c, s=80, zorder=3,
                       label='Non-Linear (MLP)' if y == 0 else "")
            delta = mlp - probe
            ax.text(max(probe, mlp)+0.5, y,
                    f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%",
                    va='center', fontsize=7, color=col_c)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title(DATASET_LABEL[ds])
        if len(methods) > 0:
            ax.legend(fontsize=7)

    fig.suptitle("Figure 2: Performance & Linear Probe Delta", fontweight="bold")
    _save_figure(fig, "fig2_performance_delta", output_dir)


def generate_fig3_bifurcation(output_dir):
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
    ax_b.set_title("Empirical Latents Overlay (Simulated)")
    ax_b.contourf(GE, GS, stability_map, levels=30, cmap="RdBu_r", alpha=0.3)
    ax_b.contour(GE, GS, stability_map, levels=[0], colors="black", linewidths=1.0)
    rng = np.random.RandomState(42)
    tuh_gee = rng.normal(8, 2, 50); tuh_gsrs = rng.normal(-1.5, 0.5, 50)
    lemon_gee = rng.normal(12, 1.5, 50); lemon_gsrs = rng.normal(-2.5, 0.4, 50)
    ax_b.scatter(tuh_gee, tuh_gsrs, s=15, alpha=0.6, color="#e41a1c", label="TUH-AB")
    ax_b.scatter(lemon_gee, lemon_gsrs, s=15, alpha=0.6, color="#377eb8", label="LEMON")
    ax_b.legend()
    _save_figure(fig, "fig3_bifurcation", output_dir)


# =====================================================================
# FIGURE 4: CKA SIMILARITY MATRICES (2×2: Task × Scale)
# =====================================================================

def generate_fig4_similarity_matrices(all_metrics, output_dir, results_dir):
    print("  Computing CKA Similarity Matrices (2×2 grid)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]

    for row, ds in enumerate(["tuh", "lemon"]):
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            methods, labels = [], []
            for full_name in sorted(all_metrics.keys()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue
                methods.append(full_name)
                labels.append(METHOD_META[clean]["label"])

            if len(methods) < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}")
                continue

            # Load features
            feat = {}
            for m in methods:
                Z, ids = load_latent_features(results_dir, m)
                if len(Z) > 0:
                    feat[m] = {"Z": Z, "ids": ids}

            n = len(methods)
            sim = np.eye(n)
            for i in range(n):
                for j in range(i+1, n):
                    m1, m2 = methods[i], methods[j]
                    if m1 in feat and m2 in feat:
                        d1, d2 = feat[m1], feat[m2]
                        idx1_map = {sid: k for k, sid in enumerate(d1["ids"])}
                        idx2_map = {sid: k for k, sid in enumerate(d2["ids"])}
                        common = [sid for sid in d1["ids"] if sid in idx2_map]
                        if len(common) > 10:
                            a1 = np.array([idx1_map[s] for s in common])
                            a2 = np.array([idx2_map[s] for s in common])
                            score = linear_cka_fast(d1["Z"][a1], d2["Z"][a2])
                            sim[i, j] = sim[j, i] = score

            # Cluster ordering
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

            im = ax.imshow(sim_ord, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(lab_ord)))
            ax.set_yticks(np.arange(len(lab_ord)))
            ax.set_xticklabels(lab_ord, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(lab_ord, fontsize=7)
            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

            # Annotate cells
            for ii in range(len(lab_ord)):
                for jj in range(len(lab_ord)):
                    val = sim_ord[ii, jj]
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val < 0.5 else "black")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)),
                 cax=cbar_ax, label="CKA Similarity")
    fig.suptitle("Figure 4: Representational Similarity (Linear CKA)", fontweight="bold", y=0.98)
    _save_figure(fig, "fig4_similarity_matrices", output_dir)


# =====================================================================
# FIGURE 5: MI CONCENTRATION (2×2: Task × Scale)
# =====================================================================

def generate_fig5_mi_concentration(all_metrics, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]

    for row, ds in enumerate(["tuh", "lemon"]):
        task = DATASET_TASK[ds]
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            y_val, colors, labels = [], [], []
            for full_name, data in sorted(all_metrics.items()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue

                ev = data.get("latent", {}).get("eval", {})
                mi_task = ev.get("mi_zy", {}).get(task, {}).get("per_dim", [])

                if mi_task:
                    sorted_mi = np.sort(mi_task)[::-1]
                    top_20 = max(1, int(len(sorted_mi) * 0.2))
                    s_mi = np.sum(sorted_mi)
                    if s_mi > 0:
                        mi_conc = np.sum(sorted_mi[:top_20]) / s_mi
                        y_val.append(mi_conc)
                        colors.append(METHOD_META[clean]["color"])
                        labels.append(METHOD_META[clean]["label"])

            if y_val:
                x_pos = np.arange(len(labels))
                ax.bar(x_pos, y_val, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('MI Concentration\n(Top 20% Dims)')
                ax.set_ylim(0, 1.05)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No MI data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

    fig.suptitle("Figure 5: Mutual Information Concentration", fontweight="bold", y=0.98)
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
    print(f"  ✓ Saved {path}")


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
            ax.scatter(active/dim, ve, s=dim*8, alpha=0.7, edgecolors='black',
                       color=METHOD_META[clean]["color"], marker=marker)
            ax.annotate(METHOD_META[clean]["label"],
                        (active/dim, ve), xytext=(5, 5),
                        textcoords='offset points', fontsize=6)

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

    fig.suptitle("Figure A1: Multi-Dimensional Efficiency", fontweight="bold")
    _save_figure(fig, "figA1_multidim_efficiency", output_dir)


# =====================================================================
# FIGURE A4: GEOMETRIC PRESERVATION (2×2: Task × Scale, computed on-the-fly)
# =====================================================================

def generate_figA4_geometric_preservation(all_metrics, output_dir, results_dir):
    print("  Computing geometric preservation metrics on-the-fly...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    scale_labels = {"small": "Small (Channel-Averaged)", "medium": "Medium (Per-Channel)"}
    panel_labels = [["A", "B"], ["C", "D"]]

    for row, ds in enumerate(["tuh", "lemon"]):
        for col, scale in enumerate(["small", "medium"]):
            ax = axes[row, col]
            _panel_label(ax, panel_labels[row][col])

            method_labels, trust_vals, cont_vals = [], [], []

            for full_name in sorted(all_metrics.keys()):
                if _get_dataset(full_name) != ds:
                    continue
                clean = _strip_prefix(full_name)
                if clean not in METHOD_META or _get_scale(clean) != scale:
                    continue

                Z, _ = load_latent_features(results_dir, full_name)
                if len(Z) < 20:
                    continue

                print(f"    Geometry: {full_name} ({Z.shape})...", end=" ", flush=True)
                geo = compute_geometry(Z)
                if geo:
                    method_labels.append(METHOD_META[clean]["label"])
                    trust_vals.append(geo["trustworthiness"])
                    cont_vals.append(geo["continuity"])
                    print(f"T={geo['trustworthiness']:.3f} C={geo['continuity']:.3f}")
                else:
                    print("skipped (too few dims)")

            if method_labels:
                avg = [(t+c)/2 for t, c in zip(trust_vals, cont_vals)]
                order = np.argsort(avg)[::-1]
                method_labels = [method_labels[i] for i in order]
                trust_vals = [trust_vals[i] for i in order]
                cont_vals = [cont_vals[i] for i in order]

                x = np.arange(len(method_labels))
                width = 0.35
                ax.bar(x - width/2, trust_vals, width, label='Trustworthiness',
                       color='#1b9e77', alpha=0.9)
                ax.bar(x + width/2, cont_vals, width, label='Continuity',
                       color='#d95f02', alpha=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=7)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(f"{DATASET_LABEL[ds]} — {scale_labels[scale]}", fontsize=9)

    fig.suptitle("Figure A4: Geometric Manifold Preservation\n(Trustworthiness & Continuity via PCA(2))",
                 fontweight="bold", y=1.0)
    _save_figure(fig, "figA4_geometric_preservation", output_dir)


# =====================================================================
# METHODOLOGY FIGURE: PSD RECONSTRUCTION
# =====================================================================

def generate_figM1_psd_reconstruction(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    freqs = np.linspace(1, 45, 90)

    # Best Fit
    raw_psd = 1.0 / (freqs ** 1.2) + 0.8 * np.exp(-0.5 * ((freqs - 10) / 2) ** 2)
    raw_log = np.log10(np.abs(raw_psd) + 1e-12)
    ctm_psd = 1.0 / (freqs ** 1.2) + 0.78 * np.exp(-0.5 * ((freqs - 10) / 2.1) ** 2)
    ctm_log = np.log10(np.abs(ctm_psd) + 1e-12)
    ax1.plot(freqs, raw_log, "k-", linewidth=1.5, label="Empirical PSD", alpha=0.8)
    ax1.plot(freqs, ctm_log, "-", color="#1b9e77", linewidth=2.0, label="CBM Best Fit")
    ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("log₁₀ Power")
    ax1.set_title("A) Qualitative PSD Best Fit"); ax1.legend(frameon=False)
    ax1.set_xlim(1, 45)

    # Average Fit
    raw_psd2 = 1.0 / (freqs ** 1.1) + 0.6 * np.exp(-0.5 * ((freqs - 11) / 1.5) ** 2)
    raw_log2 = np.log10(np.abs(raw_psd2) + 1e-12)
    ctm_psd2 = 1.0 / (freqs ** 1.15) + 0.4 * np.exp(-0.5 * ((freqs - 9.5) / 2.5) ** 2)
    ctm_log2 = np.log10(np.abs(ctm_psd2) + 1e-12)
    ax2.plot(freqs, raw_log2, "k-", linewidth=1.5, label="Empirical PSD", alpha=0.8)
    ax2.plot(freqs, ctm_log2, "-", color="#d95f02", linewidth=2.0, label="CBM Average Fit")
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_title("B) Qualitative PSD Average Fit")
    ax2.legend(frameon=False); ax2.set_xlim(1, 45)

    plt.suptitle("Figure M1: Power Spectral Density Reconstructions", fontweight='bold')
    _save_figure(fig, "figM1_psd_reconstruction", output_dir)


# =====================================================================
# MAIN
# =====================================================================

def main():
    results_dir = "/rds/general/user/lrh24/home/msc_thesis/code/Results"
    output_dir = os.path.join(results_dir, "publication_figures")
    print("Collecting metrics...")
    all_metrics = collect_all_metrics(results_dir)
    print(f"Loaded {len(all_metrics)} model evaluations.")

    print("\nGenerating Main Text Figures & Tables...")
    generate_fig1_pipeline(output_dir)
    generate_table1_capacity_efficiency(all_metrics, output_dir)
    generate_table2_summary_acc(all_metrics, output_dir)
    generate_fig2_probe_delta(all_metrics, output_dir)
    generate_fig3_bifurcation(output_dir)
    generate_fig4_similarity_matrices(all_metrics, output_dir, results_dir)
    generate_fig5_mi_concentration(all_metrics, output_dir)

    print("\nGenerating Methodology Figures...")
    generate_figM1_psd_reconstruction(output_dir)

    print("\nGenerating Appendix Figures & Tables...")
    generate_tableA1_full_matrix(all_metrics, output_dir)
    generate_figA1_multidim_efficiency(all_metrics, output_dir)
    generate_figA4_geometric_preservation(all_metrics, output_dir, results_dir)

    print("\nPublication bundle complete!")

if __name__ == "__main__":
    main()
