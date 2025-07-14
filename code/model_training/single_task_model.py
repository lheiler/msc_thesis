import torch
from torch import nn
from torch.utils.data import DataLoader
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
        hidden_dims: Tuple[int, ...] = (256, 128, 32),
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
            layers += [nn.Linear(in_f, out_f), nn.ReLU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.head = nn.Linear(last_dim, 1)

        # Losses --------------------------------------------------------
        self._bce = nn.BCELoss()
        self._mse = nn.MSELoss()

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,d) → (N,)
        h = self.trunk(x)
        out = self.head(h).squeeze(-1)  # (N,)
        return torch.sigmoid(out) if self.output_type == "classification" else out

    # -----------------------------------------------------------------
    # Helper: get criterion matching the task type
    # -----------------------------------------------------------------
    def criterion(self):
        return self._bce if self.output_type == "classification" else self._mse


# =====================================================================
#                          Training helper
# =====================================================================

def train(
    model: SingleTaskModel,
    dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: Union[str, torch.device] = "cpu",
    weight_decay: float = 0.0,
    scheduler: str = "none",
):
    """Simple training loop for a single-task model.

    Parameters follow the old ``classification_model.train`` signature for
    consistency, but only a **single** target is expected from the dataloader.
    """

    device = torch.device(device)
    model.to(device)

    try:
        weight_decay = float(weight_decay)
    except (TypeError, ValueError):
        raise ValueError(f"weight_decay must be numeric, got {weight_decay!r}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2)
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=4)
    else:
        sched = None

    criterion = model.criterion()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running = 0.0
        total = 0

        for x, y in dataloader:
            x, y = x.to(device).float(), y.to(device).float()
            optimiser.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()

            running += loss.item() * x.size(0)
            total += x.size(0)

        epoch_loss = running / max(total, 1)
        print(f"[Task-specific] Epoch {epoch:03d}: loss = {epoch_loss:.4f}")

        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(epoch_loss)
            else:
                sched.step(epoch + 1) 