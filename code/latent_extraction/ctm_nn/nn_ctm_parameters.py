"""Neural network regressor for amortised CTM parameter inference.

Maps a normalised PSD vector to an 8-element CTM parameter vector using
a simple feed-forward network trained on synthetic CTM spectra.
"""
from __future__ import annotations

from typing import Optional

import mne
import numpy as np
import torch
from torch import Tensor, nn

from utils.util import (
    PSD_CALCULATION_PARAMS,
    compute_psd_from_array,
    compute_psd_from_raw,
)


class ParameterRegressor(nn.Module):
    """MLP mapping a PSD vector to CTM parameters.

    Args:
        in_dim: Input dimension (frequency bins).  If ``None``, derived from
            the canonical Welch settings.
        hidden_dims: Hidden layer widths.
        out_dim: Output dimension (number of CTM parameters).
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        hidden_dims: tuple[int, ...] = (512, 256),
        out_dim: int = 8,
    ) -> None:
        super().__init__()
        if in_dim is None:
            _, freqs = compute_psd_from_array(
                np.zeros(
                    int(PSD_CALCULATION_PARAMS.get("n_per_seg", 256)),
                    dtype=np.float32,
                ),
                sfreq=float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0)),
                return_freqs=True,
            )
            in_dim = int(freqs.shape[0])
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def infer_latent_parameters(
    model: ParameterRegressor,
    raw: mne.io.BaseRaw,
    *,
    device: str = "cuda",
    per_channel: bool = False,
) -> np.ndarray:
    """Run the regressor on the PSD(s) of *raw*.

    Args:
        model: Trained ``ParameterRegressor``.
        raw: MNE Raw object.
        device: Torch device string.
        per_channel: If True, predict per channel; else on the average PSD.

    Returns:
        Flattened float32 parameter array.
    """
    model.eval()
    psds = compute_psd_from_raw(raw, calculate_average=not per_channel, normalize=True)
    with torch.no_grad():
        emp_input = torch.as_tensor(psds, dtype=torch.float32).to(device)
        pred = model(emp_input).cpu().numpy()
    return pred.flatten()
