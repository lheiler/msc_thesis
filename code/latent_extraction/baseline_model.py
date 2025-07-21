from __future__ import annotations

"""TorchEEG baseline feature extractor

This module provides a lightweight fallback for latent-space extraction using the
[TorchEEG](https://github.com/tczhangzhi/torcheeg) library – no training
required.  The idea is to compute **differential entropy (DE)** of the canonical
EEG frequency bands (delta, theta, alpha, beta, gamma) for each channel and
flatten the result.  The output is thus `(n_channels, n_bands)` → a 2-D matrix
flattened into a 1-D feature vector suitable for downstream MLPs.

It matches the signature expected by ``latent_extraction.extractor`` – a single
function taking an ``mne.io.Raw`` instance and returning an ``np.ndarray`` of
``dtype=float32``.
"""

from typing import Sequence
import numpy as np
import mne

# TorchEEG imports are optional – we import lazily to avoid crashing the whole
# pipeline if the dependency is missing and the user does *not* request this
# method.
try:
    from torcheeg.transforms import BandDifferentialEntropy
except ImportError as e:  # pragma: no cover – handled at call-site
    BandDifferentialEntropy = None  # type: ignore

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def extract_torcheeg(x: "mne.io.Raw", *, bands: Sequence[str] | None = None) -> np.ndarray:  # noqa: D401,E501
    """Extract **band differential entropy** features with TorchEEG.

    Parameters
    ----------
    x
        An *already-preprocessed* ``mne.io.Raw`` object (19-channel, 60-second
        clip @128 Hz in the current pipeline).
    bands
        Optional list of frequency-band names understood by TorchEEG.  If
        *None* the default bands `[delta, theta, alpha, beta, gamma]` are used.

    Returns
    -------
    numpy.ndarray
        1-D ``float32`` vector with shape ``(n_channels * n_bands,)``.
    """
    if BandDifferentialEntropy is None:  # pragma: no cover – runtime guard
        raise RuntimeError(
            "TorchEEG is not installed.  Add 'torcheeg' to your environment or "
            "omit the 'torcheeg' method from the config.")

    # ------------------------------------------------------------------
    # 1. Convert the MNE Raw object → (C, T) float32 array.
    # ------------------------------------------------------------------
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])

    data = x.get_data(picks='eeg').astype(np.float32)  # shape (C, T)

    # ------------------------------------------------------------------
    # 2. TorchEEG transform: band differential entropy
    # ------------------------------------------------------------------
    transform = BandDifferentialEntropy(bands=bands)
    feat = transform(data)  # → Tensor or ndarray (C, B)

    # Ensure *numpy* array and contiguous float32
    if not isinstance(feat, np.ndarray):
        import torch
        feat = feat.detach().cpu().numpy()

    feat = feat.astype(np.float32, copy=False)
    return feat.flatten()  # (C * B,)

