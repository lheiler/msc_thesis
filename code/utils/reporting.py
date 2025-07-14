from pathlib import Path
from typing import Dict, Any
import json
from tabulate import tabulate
import numpy as np

__all__ = ["write_markdown_report"]


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _to_builtin(obj):
    """Recursively convert NumPy scalars/arrays to Python built‑ins for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


def write_markdown_report(metrics: Dict[str, Any], output_path: str | Path):
    """Write a human-friendly Markdown report summarising evaluation metrics."""
    output_path = Path(output_path)
    lines: list[str] = ["# Model Evaluation Report\n"]

    # ------------------------------------------------------------------
    # 1. Task-specific metrics (generic) --------------------------------
    # ------------------------------------------------------------------
    if "metrics_per_task" in metrics:
        task_dict: Dict[str, Any] = metrics["metrics_per_task"]
        for task_name, task_metrics in task_dict.items():
            lines.append(_section(f"Task: {task_name}"))

            # Decide whether task is classification or regression based on keys present
            if "accuracy" in task_metrics or "loss" in task_metrics and "accuracy" in task_metrics:
                tbl = [
                    ["Loss (BCE)", task_metrics.get("loss")],
                    ["Accuracy", task_metrics.get("accuracy")],
                ]
                # We append the main metrics table *after* preparing the
                # `tbl` variable – outside the if/else – to avoid code
                # duplication.  The prediction-distribution section is
                # appended *after* that common table is written.
            else:  # regression
                tbl = [
                    ["Loss (MSE)", task_metrics.get("loss")],
                    ["MAE", task_metrics.get("mae")],
                    ["RMSE", task_metrics.get("rmse")],
                ]

            lines.append(tabulate(tbl, headers=["Metric", "Value"], tablefmt="github"))

            # ------------------------------------------------------------------
            # Optional: show how often the network predicted each label (cls)
            # ------------------------------------------------------------------
            if "pred_counts" in task_metrics:
                lines.append("\nModel prediction distribution\n")
                pred_counts = task_metrics["pred_counts"]
                n_pred = sum(pred_counts.values()) or 1  # avoid div-by-zero
                pred_tbl = [
                    [lbl, cnt, f"{cnt / n_pred * 100:.1f}%"] for lbl, cnt in pred_counts.items()
                ]
                lines.append(tabulate(pred_tbl, headers=["Label", "Count", "%"], tablefmt="github"))

    # ------------------------------------------------------------------
    # 2. Dataset stats ----------------------------------------------------
    # ------------------------------------------------------------------
    for split in ("train_dataset_stats", "eval_dataset_stats"):
        if split in metrics:
            stats = metrics[split]
            lines.append(_section(f"Dataset statistics – {split.split('_')[0]}"))
            # counts
            base_tbl = [["Samples", stats["n_samples"]]]
            lines.append(tabulate(base_tbl, tablefmt="github"))
            # gender
            n_total = stats.get("n_samples", 1)

            # ---- Gender ----
            lines.append("\nGender counts\n")
            gender_tbl = [
                [label, cnt, f"{cnt / n_total * 100:.1f}%"] for label, cnt in stats["gender_counts"].items()
            ]
            lines.append(tabulate(gender_tbl, headers=["Gender code", "Count", "%"], tablefmt="github"))

            # ---- Abnormality ----
            lines.append("\nAbnormal counts\n")
            abn_tbl = [
                [label, cnt, f"{cnt / n_total * 100:.1f}%"] for label, cnt in stats["abnormal_counts"].items()
            ]
            lines.append(tabulate(abn_tbl, headers=["Label", "Count", "%"], tablefmt="github"))
            lines.append("\nAge distribution\n")
            lines.append(tabulate(stats["age_bin_counts"].items(), headers=["Age bin", "N"], tablefmt="github"))

    # ------------------------------------------------------------------
    # 3. Independence score ---------------------------------------------
    # ------------------------------------------------------------------
    lines.append(_section("Latent-feature independence"))
    lines.append(f"Global HSIC score: **{metrics.get('global_independence_score', 'n/a')}**\n")
    lines.append("(See `hsic_matrix.png` for full matrix.)\n")

    (output_path / "final_metrics.md").write_text("\n".join(lines))

    # ------------------------------------------------------------------
    # 4. Dump raw metrics JSON ------------------------------------------
    # ------------------------------------------------------------------
    with open(output_path / "final_metrics.json", "w") as jf:
        json.dump(_to_builtin(metrics), jf, indent=2)