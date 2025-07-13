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

    # 1. Losses & global metrics
    losses_tbl = [
        ["Loss – gender (BCE)", metrics.get("loss_g")],
        ["Loss – age (MSE)", metrics.get("loss_a")],
        ["Loss – abnormality (BCE)", metrics.get("loss_abn")],
        ["Total loss", metrics.get("total_loss")],
    ]
    lines.append(_section("Losses"))
    lines.append(tabulate(losses_tbl, headers=["Metric", "Value"], tablefmt="github"))

    # 2. Classification performance
    class_tbl = [
        ["Gender accuracy", metrics.get("accuracy_g")],
        ["Abnormal accuracy", metrics.get("accuracy_abn")],
        ["Gender precision", metrics.get("gender_precision_recall_f1", {}).get("precision")],
        ["Gender recall", metrics.get("gender_precision_recall_f1", {}).get("recall")],
        ["Gender F1", metrics.get("gender_precision_recall_f1", {}).get("f1")],
        ["Abn precision", metrics.get("abn_precision_recall_f1", {}).get("precision")],
        ["Abn recall", metrics.get("abn_precision_recall_f1", {}).get("recall")],
        ["Abn F1", metrics.get("abn_precision_recall_f1", {}).get("f1")],
    ]
    lines.append(_section("Classification metrics"))
    lines.append(tabulate(class_tbl, headers=["Metric", "Value"], tablefmt="github"))

    # 3. Confusion matrices
    lines.append(_section("Confusion matrices"))
    cm_g = metrics.get("gender_confusion")
    if cm_g is not None:
        lines.append("### Gender\n")
        lines.append(tabulate(cm_g, headers=["Pred 0", "Pred 1"], showindex=["True 0", "True 1"], tablefmt="github"))
    cm_a = metrics.get("abn_confusion")
    if cm_a is not None:
        lines.append("\n### Abnormality\n")
        lines.append(tabulate(cm_a, headers=["Pred 0", "Pred 1"], showindex=["True 0", "True 1"], tablefmt="github"))

    # 4. Age regression
    lines.append(_section("Age regression"))
    age_tbl = [
        ["MAE (years)", metrics.get("mae_a")],
        ["RMSE (years)", metrics.get("rmse_a")],
    ]
    lines.append(tabulate(age_tbl, headers=["Metric", "Value"], tablefmt="github"))

    if "age_bin_mae" in metrics:
        lines.append("\n### MAE per age bin\n")
        age_bins_items = sorted(metrics["age_bin_mae"].items(), key=lambda kv: kv[0])
        lines.append(tabulate(age_bins_items, headers=["Age bin", "MAE"], tablefmt="github"))

    # 5. Dataset stats
    for split in ("train_dataset_stats", "eval_dataset_stats"):
        if split in metrics:
            stats = metrics[split]
            lines.append(_section(f"Dataset statistics – {split.split('_')[0]}"))
            # counts
            base_tbl = [["Samples", stats["n_samples"]]]
            lines.append(tabulate(base_tbl, tablefmt="github"))
            # gender
            lines.append("\nGender counts\n")
            lines.append(tabulate(stats["gender_counts"].items(), headers=["Gender code", "Count"], tablefmt="github"))
            lines.append("\nAbnormal counts\n")
            lines.append(tabulate(stats["abnormal_counts"].items(), headers=["Label", "Count"], tablefmt="github"))
            lines.append("\nAge distribution\n")
            lines.append(tabulate(stats["age_bin_counts"].items(), headers=["Age bin", "N"], tablefmt="github"))

    # 6. Independence score
    lines.append(_section("Latent-feature independence"))
    lines.append(f"Global HSIC score: **{metrics.get('global_independence_score', 'n/a')}**\n")
    lines.append("(See `hsic_matrix.png` for full matrix.)\n")

    (output_path / "final_metrics.md").write_text("\n".join(lines))

    # Also dump raw metrics JSON for programmatic use
    with open(output_path / "final_metrics.json", "w") as jf:
        json.dump(_to_builtin(metrics), jf, indent=2)