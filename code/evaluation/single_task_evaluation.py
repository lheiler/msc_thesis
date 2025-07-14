from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal, Dict, Union


def evaluate_single_task(
    model: nn.Module,
    dataloader: DataLoader,
    output_type: Literal["classification", "regression"],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Compute basic metrics for a **single** classification or regression task.

    Classification → binary-cross-entropy loss + accuracy.
    Regression     → MSE + MAE + RMSE (in *original* target units).
    """

    device = torch.device(device)
    model.to(device)
    model.eval()

    criterion = (
        nn.BCEWithLogitsLoss() if output_type == "classification" else nn.MSELoss()
    )

    total_loss = 0.0
    total = 0

    # Accumulators -----------------------------------------------------
    correct_cls = 0
    mae_sum = 0.0
    sqe_sum = 0.0
    # For classification: track how often the network predicts each label
    pred_positive = 0  # Number of times model predicts label 1 / "positive"

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device).float(), y.to(device).float()
            y_pred = model(x).squeeze(-1)

            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            if output_type == "classification":
                probs = torch.sigmoid(y_pred)
                preds = (probs > 0.5).float()
                correct_cls += (preds == y).sum().item()

                # Update prediction count statistics
                pred_positive += preds.sum().item()
            else:
                mae_sum += torch.abs(y_pred - y).sum().item()
                sqe_sum += ((y_pred - y) ** 2).sum().item()

    metrics: Dict[str, float] = {"loss": total_loss / max(total, 1)}
    if output_type == "classification":
        metrics["accuracy"] = correct_cls / max(total, 1)

        # Add label-prediction frequency (counts only – percentages rendered in report)
        pred_negative = max(total, 0) - pred_positive
        metrics["pred_counts"] = {
            "label_0": int(pred_negative),
            "label_1": int(pred_positive),
        }
    else:
        metrics["mae"] = mae_sum / max(total, 1)
        metrics["rmse"] = (sqe_sum / max(total, 1)) ** 0.5

    return metrics 