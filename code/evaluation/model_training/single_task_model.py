"""Single-task MLP model with training loop and evaluation.

Provides a lightweight fully-connected network for binary/multi-class
classification or regression on latent feature vectors, together with
a training loop supporting early stopping, learning-rate scheduling,
and optional ordinal label smoothing.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

MIN_EPOCHS_FOR_SELECTION = 5
CONFUSION_MAX_CLASSES_LABEL = 15


class SingleTaskModel(nn.Module):
    """Lightweight MLP for a single prediction target.

    Args:
        input_dim: Latent feature dimensionality.
        output_type: ``"classification"`` or ``"regression"``.
        hidden_dims: Hidden-layer widths.
        dropout: Dropout probability after each hidden layer.
        num_classes: Output units (1 = binary classification or scalar regression).
    """

    def __init__(
        self,
        input_dim: int,
        output_type: str = "classification",
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        dropout: float = 0.2,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if output_type not in {"classification", "regression"}:
            raise ValueError(
                f"output_type must be 'classification' or 'regression', got {output_type!r}"
            )
        self.output_type = output_type
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        dims = (input_dim, *hidden_dims)
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.trunk = nn.Sequential(*layers)
        self.dropout = dropout
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits / continuous output."""
        h = self.trunk(x)
        return self.head(h).squeeze(-1)

    def get_criterion(
        self, pos_weight: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """Return the appropriate loss function for this task.

        Args:
            pos_weight: Positive-class weight for binary classification.

        Returns:
            Loss module.
        """
        if self.output_type == "classification":
            if self.num_classes > 1:
                return nn.CrossEntropyLoss()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.MSELoss()

    def evaluate(
        self,
        dataloader: DataLoader,
        output_type: Optional[Literal["classification", "regression"]] = None,
        device: Union[str, torch.device] = "cpu",
        plot_dir: Optional[str] = None,
        ordinal_sigma: Optional[float] = None,
    ) -> dict[str, Any]:
        """Compute evaluation metrics on *dataloader*.

        Args:
            dataloader: Evaluation DataLoader of ``(X, y)`` batches.
            output_type: Override the model's default task type.
            device: Torch device.
            plot_dir: If set, save confusion matrix, ROC, and PR curve plots.
            ordinal_sigma: (Unused here, kept for API symmetry.)

        Returns:
            Metrics dict (loss, accuracy, precision, recall, F1, AUC, etc.).
        """
        if output_type is None:
            output_type = self.output_type

        device = torch.device(device)
        self.to(device)
        self.eval()

        criterion = (
            nn.CrossEntropyLoss()
            if (output_type == "classification" and self.num_classes > 1)
            else (nn.BCEWithLogitsLoss() if output_type == "classification" else nn.MSELoss())
        )

        total_loss = 0.0
        total = 0
        correct_cls = 0
        correct_adj = 0
        mae_sum = 0.0
        sqe_sum = 0.0

        y_true_all: list[float] = []
        y_prob_all: list[float] = []
        y_pred_all: list[float] = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device).float(), y.to(device).float()
                y_pred = self(x)

                target = (
                    y.long()
                    if (output_type == "classification" and self.num_classes > 1)
                    else y
                )
                loss = criterion(y_pred, target)
                total_loss += loss.item() * x.size(0)
                total += x.size(0)

                if output_type == "classification":
                    if self.num_classes > 1:
                        probs = torch.softmax(y_pred, dim=-1)
                        preds = torch.argmax(probs, dim=-1).float()
                        correct_adj += (torch.abs(preds - y) <= 1).sum().item()
                    else:
                        probs = torch.sigmoid(y_pred)
                        preds = (probs >= 0.5).float()
                    correct_cls += (preds == y).sum().item()

                    y_true_all.extend(y.detach().cpu().numpy().astype(float).tolist())
                    if self.num_classes == 1:
                        y_prob_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
                    y_pred_all.extend(preds.detach().cpu().numpy().astype(float).tolist())
                else:
                    mae_sum += torch.abs(y_pred - y).sum().item()
                    sqe_sum += ((y_pred - y) ** 2).sum().item()

        metrics: dict[str, Any] = {"loss": total_loss / max(total, 1)}

        if output_type == "classification":
            metrics["accuracy"] = correct_cls / max(total, 1)
            if self.num_classes > 1:
                metrics["accuracy_adj"] = correct_adj / max(total, 1)

            unique, counts = np.unique(y_pred_all, return_counts=True)
            metrics["pred_counts"] = {
                f"label_{int(k)}": int(v) for k, v in zip(unique, counts)
            }
            if self.num_classes == 1:
                metrics["pred_counts"].setdefault("label_0", 0)
                metrics["pred_counts"].setdefault("label_1", 0)

            try:
                y_true = np.asarray(y_true_all, dtype=float)
                y_pred_np = np.asarray(y_pred_all, dtype=float)

                avg_method = "macro" if self.num_classes > 1 else "binary"
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred_np, average=avg_method, zero_division=0,
                )
                metrics["precision"] = float(prec)
                metrics["recall"] = float(rec)
                metrics["f1"] = float(f1)

                if self.num_classes == 1 and len(np.unique(y_true)) == 2:
                    y_prob = np.asarray(y_prob_all, dtype=float)
                    if np.any(y_prob != y_prob[0]):
                        try:
                            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                        except ValueError:
                            pass
                        try:
                            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
                        except ValueError:
                            pass

                try:
                    cm = confusion_matrix(y_true, y_pred_np)
                    metrics["confusion"] = cm.tolist()
                except ValueError:
                    cm = None

                if plot_dir is not None:
                    os.makedirs(plot_dir, exist_ok=True)
                    self._plot_classification(
                        metrics, cm, y_true, y_prob_all, plot_dir,
                    )
            except Exception:
                pass

            logger.info("Prediction counts: %s", metrics["pred_counts"])
        else:
            metrics["mae"] = mae_sum / max(total, 1)
            metrics["rmse"] = (sqe_sum / max(total, 1)) ** 0.5
            try:
                y_true_reg, y_pred_reg = [], []
                with torch.no_grad():
                    for xb, yb in dataloader:
                        xb = xb.to(device).float()
                        yhat = self(xb)
                        y_true_reg.extend(yb.numpy().astype(float).tolist())
                        y_pred_reg.extend(yhat.cpu().numpy().astype(float).tolist())
                if len(y_true_reg) >= 2:
                    metrics["r2"] = float(r2_score(y_true_reg, y_pred_reg))
            except ValueError:
                pass

        return metrics

    def _plot_classification(
        self,
        metrics: dict[str, Any],
        cm: Optional[np.ndarray],
        y_true: np.ndarray,
        y_prob_all: list[float],
        plot_dir: str,
    ) -> None:
        """Save classification evaluation plots."""
        if cm is not None:
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.title("Confusion matrix")
            plt.colorbar()
            if self.num_classes <= CONFUSION_MAX_CLASSES_LABEL:
                ticks = np.arange(cm.shape[0])
                plt.xticks(ticks, ticks)
                plt.yticks(ticks, ticks)
            thresh = cm.max() / 2.0 if cm.size else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
            plt.close()

        if "roc_auc" in metrics and self.num_classes == 1:
            try:
                y_prob = np.asarray(y_prob_all, dtype=float)
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                plt.figure(figsize=(5, 4))
                plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
                plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
                plt.close()
            except ValueError:
                pass

        if "pr_auc" in metrics and self.num_classes == 1:
            try:
                y_prob = np.asarray(y_prob_all, dtype=float)
                precs, recs, _ = precision_recall_curve(y_true, y_prob)
                plt.figure(figsize=(5, 4))
                plt.plot(recs, precs, label=f"AP = {metrics['pr_auc']:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall curve")
                plt.legend(loc="lower left")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "pr_curve.png"))
                plt.close()
            except ValueError:
                pass


# =====================================================================
# Training loop
# =====================================================================

def train(
    model: SingleTaskModel,
    dataloader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    weight_decay: float = 0.0,
    scheduler: str = "none",
    val_split: Optional[float] = 0.2,
    random_state: int = 42,
    early_stopping_patience: int = 10,
    min_delta: float = 0.0,
    checkpoint_path: Optional[str] = None,
    plot_dir: Optional[str] = None,
    ordinal_sigma: Optional[float] = None,
) -> dict[str, Any]:
    """Train a single-task model with optional early stopping.

    Args:
        model: ``SingleTaskModel`` instance.
        dataloader: Training DataLoader.
        val_loader: Validation DataLoader (if ``None``, split from training).
        n_epochs: Maximum training epochs.
        lr: Learning rate.
        device: Torch device.
        weight_decay: L2 regularisation.
        scheduler: LR scheduler (``"plateau"``, ``"cosine"``, or ``"none"``).
        val_split: Fraction of training data to use for validation if
            *val_loader* is ``None``.
        random_state: Seed for train/val splitting.
        early_stopping_patience: Epochs without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
        checkpoint_path: If set, save best model weights here.
        plot_dir: If set, save loss/accuracy curves.
        ordinal_sigma: Gaussian sigma for ordinal label smoothing.

    Returns:
        Dict with ``best_epoch``, ``best_val_metric``, and ``history``.
    """
    device = torch.device(device)
    model.to(device)

    dataset = dataloader.dataset
    batch_size = dataloader.batch_size or 32

    if val_loader is None:
        total_samples = len(dataset)
        if val_split and 0.0 < val_split < 1.0 and total_samples >= 2:
            indices = list(range(total_samples))
            train_idx, val_idx = train_test_split(
                indices, test_size=val_split, random_state=random_state, shuffle=True,
            )
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
            logger.debug(
                "Internal split: %d train / %d val", len(train_idx), len(val_idx),
            )
        else:
            train_loader = dataloader
            val_loader = None
    else:
        train_loader = dataloader

    pos_weight = None
    if model.output_type == "classification" and model.num_classes == 1:
        all_labels = torch.cat([y for _, y in train_loader])
        num_pos = (all_labels == 1).sum().float()
        num_neg = (all_labels == 0).sum().float()
        if num_pos > 0:
            pos_weight = (num_neg / num_pos).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(weight_decay))

    sched: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=0.5, patience=early_stopping_patience,
        )

    criterion = model.get_criterion(pos_weight)

    best_metric: float = float("inf")
    best_state_dict: Optional[dict] = None
    epochs_no_improve = 0
    best_epoch = 0
    best_val_score: Optional[float] = None

    history: list[dict[str, Any]] = []
    train_losses: list[float] = []
    val_losses: list[Optional[float]] = []
    train_accs: list[float] = []
    val_accs: list[Optional[float]] = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        total = 0
        correct_cls = 0

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            optimiser.zero_grad()
            y_pred = model(x)

            if ordinal_sigma is not None and model.num_classes > 1:
                classes = torch.arange(model.num_classes, device=device).float()
                target = torch.exp(
                    -0.5 * ((classes.view(1, -1) - y.view(-1, 1)) / ordinal_sigma) ** 2
                )
                target = target / target.sum(dim=-1, keepdim=True)
                loss = criterion(y_pred, target)
            else:
                target = (
                    y.long()
                    if (model.output_type == "classification" and model.num_classes > 1)
                    else y
                )
                loss = criterion(y_pred, target)

            loss.backward()
            optimiser.step()
            running += loss.item() * x.size(0)
            total += x.size(0)

            if model.output_type == "classification":
                if model.num_classes > 1:
                    preds = torch.argmax(y_pred, dim=-1)
                else:
                    preds = (torch.sigmoid(y_pred) >= 0.5).float()
                correct_cls += (preds == y).sum().item()

        epoch_loss = running / max(total, 1)
        epoch_acc = correct_cls / max(total, 1) if model.output_type == "classification" else None

        val_loss: Optional[float] = None
        val_acc: Optional[float] = None
        current_val_score: Optional[float] = None

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device).float(), yb.to(device).float()
                    y_pred = model(xb)
                    target = (
                        yb.long()
                        if (model.output_type == "classification" and model.num_classes > 1)
                        else yb
                    )
                    v_loss = criterion(y_pred, target)
                    val_running += v_loss.item() * xb.size(0)
                    val_total += xb.size(0)
                    if model.output_type == "classification":
                        if model.num_classes > 1:
                            preds = torch.argmax(y_pred, dim=-1)
                        else:
                            preds = (torch.sigmoid(y_pred) >= 0.5).float()
                        val_correct += (preds == yb).sum().item()

            val_loss = val_running / max(val_total, 1)
            if model.output_type == "classification":
                val_acc = val_correct / max(val_total, 1)
                current_val_score = val_acc

        if epoch % 10 == 0:
            msg = f"loss={epoch_loss:.4f}"
            if epoch_acc is not None:
                msg += f" acc={epoch_acc:.2f}"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.4f}"
            if val_acc is not None:
                msg += f" val_acc={val_acc:.2f}"
            logger.info("Epoch %03d: %s", epoch, msg)

        current_metric = val_loss if val_loss is not None else float("inf")
        if epoch >= MIN_EPOCHS_FOR_SELECTION and (current_metric + min_delta < best_metric):
            best_metric = float(current_metric)
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_score = current_val_score
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1

        entry: dict[str, Any] = {"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss}
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        if model.output_type == "classification":
            entry.update({"train_acc": epoch_acc, "val_acc": val_acc})
            train_accs.append(epoch_acc or 0.0)
            val_accs.append(val_acc)
        history.append(entry)

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            logger.info(
                "Early stopping after %d epochs (best=%d, val_loss=%.4f)",
                epoch, best_epoch, best_metric,
            )
            break

        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_loss if val_loss is not None else epoch_loss)
            else:
                sched.step(epoch + 1)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if plot_dir is not None:
        _plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            model.output_type, plot_dir,
        )

    return {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_val_score": best_val_score,
        "history": history,
    }


def _plot_training_curves(
    train_losses: list[float],
    val_losses: list[Optional[float]],
    train_accs: list[float],
    val_accs: list[Optional[float]],
    output_type: str,
    plot_dir: str,
) -> None:
    """Save loss and accuracy training curves to *plot_dir*."""
    os.makedirs(plot_dir, exist_ok=True)
    epochs_range = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, label="Train")
    if any(v is not None for v in val_losses):
        plt.plot(
            epochs_range,
            [v if v is not None else float("nan") for v in val_losses],
            label="Val",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()

    if output_type == "classification" and train_accs:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_range, train_accs, label="Train")
        if any(v is not None for v in val_accs):
            plt.plot(
                epochs_range,
                [v if v is not None else float("nan") for v in val_accs],
                label="Val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training accuracy curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "accuracy_curve.png"))
        plt.close()
