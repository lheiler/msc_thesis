import torch
from torch import nn
from typing import Union, Tuple

class TransformerTaskModel(nn.Module):
    """Single-task predictor using a lightweight Transformer encoder as the head.

    The input is expected to be a **flattened** latent vector of size
        ``input_dim = n_channels * d_model``.
    Internally we reshape to ``(B, n_channels, d_model)`` and apply an
    ``nn.TransformerEncoder``.

    Parameters
    ----------
    input_dim : int
        Flattened latent dimensionality (must equal ``n_channels * d_model``).
    output_type : {"classification", "regression"}
        Decides activation / loss.
    n_channels : int, optional
        Number of EEG channels that were preserved by the encoder.
    d_model : int, optional
        Feature size (embedding dimension) per channel produced by the encoder.
    n_head : int, optional
        Number of attention heads in each Transformer layer.
    n_layer : int, optional
        Depth (stacked encoder layers).
    dropout : float, optional
        Dropout prob applied after Transformer (None inside Transformer).
    """

    def __init__(
        self,
        input_dim: int,
        output_type: str = "classification",
        *,
        n_channels: int = 19,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if output_type not in {"classification", "regression"}:
            raise ValueError(
                f"output_type must be 'classification' or 'regression', got {output_type!r}"
            )
        if input_dim != n_channels * d_model:
            raise ValueError(
                "input_dim must equal n_channels * d_model. "
                f"Got input_dim={input_dim}, n_channels={n_channels}, d_model={d_model}."
            )

        self.output_type = output_type
        self.n_channels = n_channels
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.flatten = nn.Flatten()
        self.head = nn.Linear(n_channels * d_model, 1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B,)
        if x.dim() != 2:
            raise ValueError(
                "Expected 2-D input tensor of shape (batch, input_dim).")

        b, dim = x.shape
        if dim != self.n_channels * self.d_model:
            raise ValueError(
                f"Input feature dimension mismatch. Expected {self.n_channels * self.d_model}, got {dim}.")

        # Reshape to (B, n_channels, d_model)
        z = x.view(b, self.n_channels, self.d_model)
        z = self.transformer(z)
        z = self.dropout(z)
        z = self.flatten(z)
        out = self.head(z).squeeze(-1)  # (B,)
        return out  # logits for classification, value for regression

    # ------------------------------------------------------------------
    # Loss helper (mirrors SingleTaskModel)
    # ------------------------------------------------------------------
    def get_criterion(self, pos_weight: Union[torch.Tensor, None] = None):
        if self.output_type == "classification":
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.MSELoss() 