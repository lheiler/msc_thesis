"""Markdown and JSON report generation for evaluation results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from tabulate import tabulate

__all__ = ["write_markdown_report"]


def _section(title: str) -> str:
    """Return a Markdown level-2 heading."""
    return f"\n## {title}\n"


def _to_builtin(obj: Any) -> Any:
    """Recursively convert NumPy types to Python built-ins for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    return obj


def write_markdown_report(
    metrics: dict[str, Any], output_path: Union[str, Path]
) -> None:
    """Write a human-friendly Markdown report of evaluation metrics.

    Args:
        metrics: Nested metrics dictionary from the pipeline.
        output_path: Directory where ``final_metrics.md`` and
            ``final_metrics.json`` will be saved.
    """
    output_path = Path(output_path)
    lines: list[str] = ["# Model Evaluation Report\n"]

    if "metrics_per_task" in metrics:
        task_dict: dict[str, Any] = metrics["metrics_per_task"]
        for task_name, task_metrics in task_dict.items():
            lines.append(_section(f"Task: {task_name}"))

            if "accuracy" in task_metrics:
                tbl = [
                    ["Loss (BCE)", task_metrics.get("loss")],
                    ["Accuracy", task_metrics.get("accuracy")],
                ]
            else:
                tbl = [
                    ["Loss (MSE)", task_metrics.get("loss")],
                    ["MAE", task_metrics.get("mae")],
                    ["RMSE", task_metrics.get("rmse")],
                ]

            lines.append(tabulate(tbl, headers=["Metric", "Value"], tablefmt="github"))

            if "pred_counts" in task_metrics:
                lines.append("\nModel prediction distribution\n")
                pred_counts = task_metrics["pred_counts"]
                n_pred = sum(pred_counts.values()) or 1
                pred_tbl = [
                    [lbl, cnt, f"{cnt / n_pred * 100:.1f}%"]
                    for lbl, cnt in pred_counts.items()
                ]
                lines.append(
                    tabulate(pred_tbl, headers=["Label", "Count", "%"], tablefmt="github")
                )

    for split in ("train_dataset_stats", "eval_dataset_stats"):
        if split not in metrics:
            continue
        stats = metrics[split]
        lines.append(_section(f"Dataset statistics - {split.split('_')[0]}"))
        lines.append(tabulate([["Samples", stats["n_samples"]]], tablefmt="github"))

        n_total = stats.get("n_samples", 1)
        lines.append("\nGender counts\n")
        gender_tbl = [
            [label, cnt, f"{cnt / n_total * 100:.1f}%"]
            for label, cnt in stats["gender_counts"].items()
        ]
        lines.append(
            tabulate(gender_tbl, headers=["Gender code", "Count", "%"], tablefmt="github")
        )

        lines.append("\nAbnormal counts\n")
        abn_tbl = [
            [label, cnt, f"{cnt / n_total * 100:.1f}%"]
            for label, cnt in stats["abnormal_counts"].items()
        ]
        lines.append(
            tabulate(abn_tbl, headers=["Label", "Count", "%"], tablefmt="github")
        )

        lines.append("\nAge distribution\n")
        lines.append(
            tabulate(stats["age_bin_counts"].items(), headers=["Age bin", "N"], tablefmt="github")
        )

    lines.append(_section("Latent-feature independence"))
    lines.append(
        f"Global HSIC score: **{metrics.get('global_independence_score', 'n/a')}**\n"
    )
    lines.append("(See `hsic_matrix.png` for full matrix.)\n")

    (output_path / "final_metrics.md").write_text("\n".join(lines))

    with open(output_path / "final_metrics.json", "w") as jf:
        json.dump(_to_builtin(metrics), jf, indent=2)
