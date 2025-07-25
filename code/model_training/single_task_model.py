import torch
from torch import nn
# --- Additional utilities for validation split ---
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from typing import Tuple, Union


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
        Decides final activation as well as default loss (BCE vs. MSE).
    hidden_dims : tuple[int, ...]
        Sizes of hidden layers for the MLP trunk.
    dropout : float
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_type: str = "classification",
        hidden_dims: Tuple[int, ...] = (512, 256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if output_type not in {"classification", "regression"}:
            raise ValueError(
                f"output_type must be 'classification' or 'regression', got {output_type!r}"
            )
        self.output_type = output_type

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
        self.head = nn.Linear(last_dim, 1)

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
            return nn.BCEWithLogitsLoss()
        else:
            return nn.MSELoss()


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
):
    """Simple training loop for a single-task model.

    Parameters follow the old ``classification_model.train`` signature for
    consistency, but only a **single** target is expected from the dataloader.
    """
    device = torch.device(device)
    model.to(device)

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
            print(f"📑 Split {total_samples} samples → {len(train_subset)} train | {len(val_subset)} val")
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
    if model.output_type == "classification":
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

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        total = 0
        correct_cls = 0  # correct predictions for classification tasks

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            optimiser.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            #print(torch.sigmoid(y_pred).detach().cpu().numpy())
            running += loss.item() * x.size(0)
            total += x.size(0)

            # ---------------------------------------------------------
            # Track accuracy for *classification* tasks only
            # ---------------------------------------------------------
            if model.output_type == "classification":
                probs = torch.sigmoid(y_pred)
                preds = (probs >= 0.5).float()
                #print(preds)
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
                    v_loss = criterion(y_pred, yb)
                    val_running += v_loss.item() * xb.size(0)
                    val_total += xb.size(0)

                    if model.output_type == "classification":
                        preds = (torch.sigmoid(y_pred) >= 0.5).float()
                        val_correct_cls += (preds == yb).sum().item()

            val_loss = val_running / max(val_total, 1)
            if model.output_type == "classification":
                val_acc = val_correct_cls / max(val_total, 1)
                msg += f" | val_loss = {val_loss:.4f}, val_acc = {val_acc:.2f}"
                current_val_score = val_acc  # higher better for acc
                plateau_metric = val_loss    # still use loss for early-stopping
            else:
                msg += f" | val_loss = {val_loss:.4f}"
                current_val_score = -val_loss  # lower loss ⇒ higher score
                plateau_metric = val_loss
        else:
            val_loss = None
            current_val_score = None
            plateau_metric = epoch_loss

        if epoch % 1 == 0: print(f"[Task-specific] Epoch {epoch:03d}: {msg}")

        # ---------------- Early-stopping tracking ------------------
        current_metric = plateau_metric
        if current_metric + min_delta < best_metric:
            best_metric = current_metric
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_score = current_val_score
            epochs_no_improve = 0
            if checkpoint_path is not None:
                torch.save(best_state_dict, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Keep simple history record
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
        })

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(
                f"⏹️  Early stopping after {epoch} epochs (best epoch {best_epoch}, "
                f"val_metric={best_metric:.4f})."
            )
            break

        # ---------------- Scheduler step -----------------------------
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(plateau_metric)
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

    # Optional: return basic info – keeps backward compatibility (None ignored by caller)
    return {
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "best_val_score": best_val_score if "best_val_score" in locals() else None,
        "history": history,
    }