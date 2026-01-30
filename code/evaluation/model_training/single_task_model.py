import torch
from torch import nn
# --- Additional utilities for validation split ---
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Union, Literal, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt  # For training curves
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    r2_score,
    roc_curve,
    precision_recall_curve,
)


class SingleTaskModel(nn.Module):
    """Lightweight MLP for **one** prediction target (classification or regression).

    The architecture matches the former *shared trunk* used in ``ClassificationModel``
    but is now fully *independent* per task ‑ i.e. **no parameter sharing** between
    different targets.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the latent feature vector.
    output_type : {"classification", "regression"}
        Decides final activation as well as default loss (BCE/CE vs. MSE).
    hidden_dims : tuple[int, ...]
        Sizes of hidden layers for the MLP trunk.
    dropout : float
        Dropout probability applied after every hidden layer.
    num_classes : int, optional
        Number of output units. Default is 1 (binary classification or regression).
        If > 1, treating as multi-class classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_type: str = "classification",
        hidden_dims: Tuple[int, ...] = (512, 256, 128, 64),
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

        # ---------------------------- trunk ----------------------------
        layers = []
        dims = (input_dim, *hidden_dims)
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.trunk = nn.Sequential(*layers)
        self.dropout = dropout

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, num_classes)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,d) → (N,)
        h = self.trunk(x)
        # Return **raw logits** for classification tasks – the calling code is
        # responsible for applying ``torch.sigmoid`` when probabilities are
        # required (e.g. during evaluation).
        out = self.head(h).squeeze(-1)  # (N,)
        return out  # logits for classification, continuous output for regression

    # -----------------------------------------------------------------
    # Helper: get criterion matching the task type
    # -----------------------------------------------------------------
    def get_criterion(self, pos_weight: torch.Tensor = None):
        if self.output_type == "classification":
            if self.num_classes > 1:
                return nn.CrossEntropyLoss()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.MSELoss()

    # -----------------------------------------------------------------
    # Evaluation helper (moved from evaluation/single_task_evaluation.py)
    # -----------------------------------------------------------------
    def evaluate(
        self,
        dataloader: DataLoader,
        output_type: Literal["classification", "regression"] | None = None,
        device: Union[str, torch.device] = "cpu",
        plot_dir: str | None = None,
        ordinal_sigma: float | None = None,
    ) -> Dict[str, Any]:
        """Compute metrics on *dataloader* using *this* model.

        Classification → BCE loss + accuracy (+ prediction counts).
        Regression     → MSE + MAE + RMSE.

        The implementation mirrors the former ``evaluate_single_task`` helper but
        lives inside the model class to guarantee **one single source of truth**.
        """

        if output_type is None:
            output_type = self.output_type

        device = torch.device(device)
        self.to(device)
        self.eval()

        criterion = (
            nn.CrossEntropyLoss() if (output_type == "classification" and self.num_classes > 1) 
            else (nn.BCEWithLogitsLoss() if output_type == "classification" else nn.MSELoss())
        )

        total_loss = 0.0
        total = 0

        # Accumulators -------------------------------------------------
        correct_cls = 0
        correct_adj = 0 # Off-by-one accuracy for ordinal tasks
        mae_sum = 0.0
        sqe_sum = 0.0
        pred_positive = 0  # Number of times model predicts label 1 / "positive"

        # For rich metrics/plots
        y_true_all: list[float] = []
        y_prob_all: list[float] = []
        y_pred_all: list[float] = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device).float(), y.to(device).float()
                y_pred = self(x)

                target = y.long() if (output_type == "classification" and self.num_classes > 1) else y
                loss = criterion(y_pred, target)
                total_loss += loss.item() * x.size(0)
                total += x.size(0)

                if output_type == "classification":
                    if self.num_classes > 1:
                        probs = torch.softmax(y_pred, dim=-1)
                        preds = torch.argmax(probs, dim=-1).float()
                        
                        # Adjacent accuracy (Top-2 adjacent)
                        if self.num_classes > 1:
                            correct_adj += (torch.abs(preds - y) <= 1).sum().item()
                    else:
                        probs = torch.sigmoid(y_pred)
                        preds = (probs >= 0.5).float()
                    
                    correct_cls += (preds == y).sum().item()
                    pred_positive += preds.sum().item()

                    # collect for plots/metrics
                    y_true_all.extend(y.detach().cpu().numpy().astype(float).tolist())
                    if self.num_classes == 1:
                        y_prob_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
                    y_pred_all.extend(preds.detach().cpu().numpy().astype(float).tolist())
                else:
                    mae_sum += torch.abs(y_pred - y).sum().item()
                    sqe_sum += ((y_pred - y) ** 2).sum().item()

        metrics: Dict[str, Any] = {"loss": total_loss / max(total, 1)}
        if output_type == "classification":
            metrics["accuracy"] = correct_cls / max(total, 1)
            if self.num_classes > 1:
                metrics["accuracy_adj"] = correct_adj / max(total, 1)
            
            pred_negative = max(total, 0) - pred_positive
            metrics["pred_counts"] = {
                "label_0": int(pred_negative),
                "label_1": int(pred_positive),
            }

            # Additional classification metrics
            try:
                y_true = np.asarray(y_true_all, dtype=float)
                y_pred = np.asarray(y_pred_all, dtype=float)

                # precision/recall/f1
                avg_method = "macro" if self.num_classes > 1 else "binary"
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average=avg_method, zero_division=0
                )
                metrics["precision"] = float(prec)
                metrics["recall"] = float(rec)
                metrics["f1"] = float(f1)

                # ROC-AUC and PR-AUC when feasible (binary or explicitly handled multiclass)
                if self.num_classes == 1:
                    if len(np.unique(y_true)) == 2 and np.any(y_prob_all != y_prob_all[0]):
                        y_prob = np.asarray(y_prob_all, dtype=float)
                        try:
                            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                        except Exception: pass
                        try:
                            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
                        except Exception: pass

                # Confusion matrix
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    metrics["confusion"] = cm.tolist()
                except Exception:
                    pass

                # Plots
                if plot_dir is not None:
                    os.makedirs(plot_dir, exist_ok=True)

                    # Confusion matrix heatmap
                    if "confusion" in metrics:
                        plt.figure(figsize=(4, 4))
                        plt.imshow(cm, interpolation="nearest", cmap="Blues")
                        plt.title("Confusion matrix")
                        plt.colorbar()
                        tick_marks = np.arange(self.num_classes)
                        if self.num_classes <= 15: # Only label if readable
                             plt.xticks(tick_marks, tick_marks)
                             plt.yticks(tick_marks, tick_marks)
                        thresh = cm.max() / 2.0 if cm.size else 0.5
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                plt.text(j, i, format(cm[i, j], "d"),
                                         ha="center", va="center",
                                         color="white" if cm[i, j] > thresh else "black")
                        plt.ylabel("True label")
                        plt.xlabel("Predicted label")
                        plt.tight_layout()
                        plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
                        plt.close()

                    # ROC curve
                    if "roc_auc" in metrics:
                        try:
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
                        except Exception:
                            pass

                    # Precision-Recall curve
                    if "pr_auc" in metrics:
                        try:
                            precs, recs, _ = precision_recall_curve(y_true, y_prob)
                            plt.figure(figsize=(5, 4))
                            plt.plot(recs, precs, label=f"AP = {metrics['pr_auc']:.3f}")
                            plt.xlabel("Recall")
                            plt.ylabel("Precision")
                            plt.title("Precision–Recall curve")
                            plt.legend(loc="lower left")
                            plt.tight_layout()
                            plt.savefig(os.path.join(plot_dir, "pr_curve.png"))
                            plt.close()
                        except Exception:
                            pass

            except Exception:
                # keep core metrics even if extras fail
                pass

            print("🔍 Prediction counts:", metrics["pred_counts"])
        else:
            metrics["mae"] = mae_sum / max(total, 1)
            metrics["rmse"] = (sqe_sum / max(total, 1)) ** 0.5
            # R^2 for regression when labels vary
            try:
                # Need to re-run pass to gather predictions for R^2 without duplicating lots of code
                y_true_all_reg, y_pred_all_reg = [], []
                with torch.no_grad():
                    for xb, yb in dataloader:
                        xb = xb.to(device).float()
                        yb = yb.to(device).float()
                        yhat = self(xb)
                        y_true_all_reg.extend(yb.detach().cpu().numpy().astype(float).tolist())
                        y_pred_all_reg.extend(yhat.detach().cpu().numpy().astype(float).tolist())
                if len(y_true_all_reg) >= 2:
                    metrics["r2"] = float(r2_score(y_true_all_reg, y_pred_all_reg))
            except Exception:
                pass

        return metrics


# =====================================================================
#                          Training helper
# =====================================================================

def train(
    model: SingleTaskModel,
    dataloader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    n_epochs: int = 100,
    lr: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    weight_decay: float = 0.0,
    scheduler: str = "none",
    val_split: float = 0.2,
    random_state: int = 42,
    early_stopping_patience: int = 10,
    min_delta: float = 0.0,
    checkpoint_path: str | None = None,
    plot_dir: str | None = None,
    ordinal_sigma: float | None = None,
):
    """Simple training loop for a single-task model.

    Parameters follow the old ``classification_model.train`` signature for
    consistency, but only a **single** target is expected from the dataloader.
    """
    device = torch.device(device)

    # --- Stability knobs (embedded defaults, no config bloat) ---
    _MIN_EPOCHS_FOR_SELECTION = 5   # don't accept a best checkpoint before this epoch

    model.to(device)

    print(f"Training model with checkpoint path: {checkpoint_path}")
    # -----------------------------------------------------------------
    # 0. Derive *internal* train / validation loaders ------------------
    # If a validation loader is provided externally we skip splitting.
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size or 32

    if val_loader is None:
        total_samples = len(dataset)
        if 0.0 < val_split < 1.0 and total_samples >= 2:
            indices = list(range(total_samples))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_split,
                random_state=random_state,
                shuffle=True,
            )

            train_subset = Subset(dataset, train_idx)
            val_subset   = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
            print(f"📑OH OHH; CAREFULL CREATING SPLIT in the wrong place; split {total_samples} samples → {len(train_subset)} train | {len(val_subset)} val")
        else:
            train_loader = dataloader
            val_loader   = None
    else:
        train_loader = dataloader  # use given loader as train
        # Ensure provided val_loader batch_size defined
        # (no action needed)

    # -----------------------------------------------------------------
    # 1. Loss / optimiser / scheduler setup ---------------------------
    # -----------------------------------------------------------------
    try:
        weight_decay = float(weight_decay)
    except (TypeError, ValueError):
        raise ValueError(f"weight_decay must be numeric, got {weight_decay!r}")

    pos_weight = None
    if model.output_type == "classification" and model.num_classes == 1:
        all_labels = torch.cat([y for _, y in train_loader])
        num_pos = (all_labels == 1).sum().float()
        num_neg = (all_labels == 0).sum().float()
        if num_pos > 0:
            pos_weight = (num_neg / num_pos).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=early_stopping_patience)
    else:
        sched = None

    criterion = model.get_criterion(pos_weight)

    # -----------------------------------------------------------------
    # 2. Early-stopping initialisation
    # -----------------------------------------------------------------
    best_metric: float = float("inf")  # lower = better
    best_state_dict = None
    epochs_no_improve = 0
    best_epoch = 0

    history = []  # store per-epoch metrics if caller is interested
    # Keep separate lists for quick plotting
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []  # may stay empty for regression

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        total = 0
        correct_cls = 0  # correct predictions for classification tasks

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            optimiser.zero_grad()
            y_pred = model(x)
            
            # ---------------------------------------------------------
            # Ordinal Label Smoothing (Gaussian)
            # ---------------------------------------------------------
            if ordinal_sigma is not None and model.num_classes > 1:
                # Create a target distribution instead of a single index
                # y is the "true" index [0, num_classes-1]
                classes = torch.arange(model.num_classes).to(device).float()
                # Gaussian kernel: exp(-0.5 * ((x-mu)/sigma)^2)
                # target dist: (N, C)
                target = torch.exp(-0.5 * ((classes.view(1, -1) - y.view(-1, 1)) / ordinal_sigma)**2)
                target = target / target.sum(dim=-1, keepdim=True)
                loss = criterion(y_pred, target)
            else:
                # CrossEntropy expects long labels (N) if not soft, MSE/BCE expect floats (N)
                # CE can handle (N, C) targets for soft labels in recent PyTorch
                target = y.long() if (model.output_type == "classification" and model.num_classes > 1) else y
                loss = criterion(y_pred, target)
            
            loss.backward()
            optimiser.step()
            #print(torch.sigmoid(y_pred).detach().cpu().numpy())
            running += loss.item() * x.size(0)
            total += x.size(0)

            # ---------------------------------------------------------
            # Track accuracy for *classification* tasks only
            # ---------------------------------------------------------
            if model.output_type == "classification":
                if model.num_classes > 1:
                    preds = torch.argmax(y_pred, dim=-1)
                else:
                    preds = (torch.sigmoid(y_pred) >= 0.5).float()
                correct_cls += (preds == y).sum().item()

        epoch_loss = running / max(total, 1)

        # ---------------- Metrics ------------------------------------
        
        if model.output_type == "classification":
            epoch_acc = correct_cls / max(total, 1)
            msg = f"loss = {epoch_loss:.4f}, acc = {epoch_acc:.2f}"
        else:
            msg = f"loss = {epoch_loss:.4f}"

        # -------- Validation evaluation (optional) -------------------
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_total = 0
            val_correct_cls = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device).float(), yb.to(device).float()
                    y_pred = model(xb)
                    
                    target = yb.long() if (model.output_type == "classification" and model.num_classes > 1) else yb
                    v_loss = criterion(y_pred, target)
                    
                    val_running += v_loss.item() * xb.size(0)
                    val_total += xb.size(0)

                    if model.output_type == "classification":
                        if model.num_classes > 1:
                            preds = torch.argmax(y_pred, dim=-1)
                        else:
                            preds = (torch.sigmoid(y_pred) >= 0.5).float()
                        val_correct_cls += (preds == yb).sum().item()

            val_loss = val_running / max(val_total, 1)
            if model.output_type == "classification":
                val_acc = val_correct_cls / max(val_total, 1)
                msg += f" | val_loss = {val_loss:.4f}, val_acc = {val_acc:.2f}"
                # Always use val_loss for scheduler & model selection; val_acc for logging
                current_val_score = val_acc      # for logging/reporting only
                plateau_metric = val_loss        # scheduler & early stopping on loss
            else:
                msg += f" | val_loss = {val_loss:.4f}"
                current_val_score = -val_loss  # lower loss ⇒ higher score (for reporting)
                plateau_metric = val_loss
        else:
            val_loss = None
            val_acc  = None

        if epoch % 1 == 0: print(f"[Task-specific] Epoch {epoch:03d}: {msg}")

        # ---------------- Early-stopping tracking ------------------
        # Use validation loss for selection; require a minimum number of epochs
        current_metric = val_loss if val_loss is not None else float("inf")
        if epoch >= _MIN_EPOCHS_FOR_SELECTION and (current_metric + min_delta < best_metric):
            best_metric = float(current_metric)
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_score = current_val_score if 'current_val_score' in locals() else None
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1

        # ---------------- History bookkeeping -------------------
        entry = {"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss}
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if model.output_type == "classification":
            entry.update({"train_acc": epoch_acc, "val_acc": val_acc})
            train_accs.append(epoch_acc)
            val_accs.append(val_acc)

        history.append(entry)

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(
                f"⏹️  Early stopping after {epoch} epochs (best epoch {best_epoch}, "
                f"best_val_loss={best_metric:.4f})."
            )
            break

        # ---------------- Scheduler step -----------------------------
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # step on validation loss (lower is better); if no val loader, fall back to train loss
                sched.step(val_loss if val_loss is not None else epoch_loss)
            else:
                sched.step(epoch + 1)

        # Quick distribution check on training predictions ------------
        model.eval()
        all_logits = []
        for xb, _ in train_loader:
            xb = xb.to(device).float()
            all_logits.append(model(xb))
        logits = torch.cat(all_logits)
        preds  = (torch.sigmoid(logits) >= 0.5).float()
        #print("epoch", epoch, "train-pred-dist:", (preds == 0).sum().item(), (preds == 1).sum().item())
        model.train()

    # -----------------------------------------------------------------
    # Restore best weights at end of training
    # -----------------------------------------------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # -----------------------------------------------------------------
    #  Post-training plots ---------------------------------------------
    # -----------------------------------------------------------------
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

        epochs_range = list(range(1, len(train_losses) + 1))

        # Loss curve ---------------------------------------------------
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_range, train_losses, label="Train")
        if any(v is not None for v in val_losses):
            plt.plot(epochs_range, [v if v is not None else float('nan') for v in val_losses], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss curve")
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(plot_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()

        # Accuracy curve (classification only) ------------------------
        if model.output_type == "classification" and train_accs:
            plt.figure(figsize=(6, 4))
            plt.plot(epochs_range, train_accs, label="Train")
            if any(v is not None for v in val_accs):
                plt.plot(epochs_range, [v if v is not None else float('nan') for v in val_accs], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training accuracy curve")
            plt.legend()
            plt.tight_layout()
            acc_path = os.path.join(plot_dir, "accuracy_curve.png")
            plt.savefig(acc_path)
            plt.close()

    # Optional: return basic info – keeps backward compatibility (None ignored by caller)
    return {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_val_score": best_val_score if "best_val_score" in locals() else None,
        "history": history,
    }