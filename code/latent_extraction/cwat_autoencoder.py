"""CwA-T auto-encoder latent extraction.

This module provides a single public function
    extract_cwat(signals, device="cuda"|"cpu")
which can be plugged into the generic `latent_extraction.extractor` pipeline.

The function lazily loads the *encoder* part of the CwA-T network using the
weights stored by the training script (either ``best.pth`` or
``optuna_best.pth``).  To minimise disk usage, **only the current best model
is kept** – the training code deletes older checkpoints automatically.

Supported usage from ``extractor.py``::

    latent = extract_cwat(batch_signals)  # returns (B, n_channels*d_model)

The first dimension is the batch size.  The returned tensor lives on CPU so it
can be serialised safely.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn


# -----------------------------------------------------------------------------
#  Import CwA-T modules (encoder variants)
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_DIR.parents[2]  # /…/code/

# The CwA-T implementation lives in extras/models/CwA-T
_CWAT_DIR = _PROJECT_ROOT / "extras" / "models" / "CwA-T"
sys.path.append(str(_CWAT_DIR))

# Encoder variants (res_encoderS / M / L)
from models.encoder import res_encoderS, res_encoderM, res_encoderL  # type: ignore


__all__ = ["extract_cwat"]

# Public accessor for early loading (used by extractor.py)
def get_cwat_model(*, device: str | torch.device | None = None) -> nn.Module:
    """Return the cached encoder instance, loading it if necessary."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else (
                torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            )
        )
    elif isinstance(device, str):
        device = torch.device(device)

    _MODEL = _load_encoder(device)
    return _MODEL


# -----------------------------------------------------------------------------
#  Global cache so that the model is loaded only once
# -----------------------------------------------------------------------------

_MODEL: Optional[nn.Module] = None


def _find_best_weights() -> Path:
    """Return the path to *optuna_best.pth* (preferred) or *best.pth*.

    Searches under ``extras/models/weights/**`` – matching how the training
    script constructs its checkpoint directory.  Raises ``FileNotFoundError``
    if no suitable file is discovered.
    """

    weights_root = _PROJECT_ROOT / "extras" / "models" / "weights"
    if not weights_root.exists():
        raise FileNotFoundError(
            "Cannot locate the 'weights' directory – have you trained the "
            "CwA-T model yet?"
        )

    # Search depth-first for optuna_best.pth → best.pth
    patterns: List[Tuple[str, str]] = [
        ("optuna_best.pth", "**/optuna_best.pth"),
        ("best.pth", "**/best.pth"),
    ]

    for _name, glob_pat in patterns:
        matches = list(weights_root.glob(glob_pat))
        if matches:
            # Return the *latest* file if multiple exist (sorted by mtime)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    raise FileNotFoundError(
        "No suitable CwA-T weight file found (optuna_best.pth / best.pth)."
    )


def _load_encoder(device: torch.device) -> nn.Module:
    """Instantiate the encoder and load the saved weights.

    Because hyper-parameters like ``d_model`` and ``n_head`` can vary across
    Optuna trials, we try a set of common configurations until the state-dict
    loads without shape mismatches.
    """

    state_dict_path = _find_best_weights()
    state_dict = torch.load(state_dict_path, map_location="cpu")

    candidate_d_models = [128, 256, 384]
    candidate_heads = [2, 4, 8]
    candidate_variants = ["L", "M", "S"]  # search order: larger → smaller

    n_channels = 19  # fixed for TUH / Harvard datasets

    for variant in candidate_variants:
        for d_model in candidate_d_models:
            try:
                if variant == "L":
                    enc_fn = res_encoderL
                elif variant == "M":
                    enc_fn = res_encoderM
                else:
                    enc_fn = res_encoderS

                encoder = enc_fn(
                    n_channels=n_channels,
                    groups=n_channels,
                    num_classes=2,  # unused; kept for signature compatibility
                    d_model=d_model,
                )

                # Load *strict* so mismatched shapes raise immediately
                encoder.load_state_dict(state_dict, strict=True)

                encoder.eval()
                encoder.to(device)

                print(
                    f"[CwA-T] Loaded encoder variant={variant} d_model={d_model} "
                    f"from {state_dict_path.relative_to(_PROJECT_ROOT)}"
                )

                return encoder
            except (RuntimeError, KeyError):
                # Shape mismatch → try next candidate
                continue

    raise RuntimeError(
        "Could not match the saved weights with any known encoder variant." (
            "Weights path: " + str(state_dict_path)
        )
    )


def extract_cwat(signals: Tensor, *, device: str | torch.device | None = None) -> Tensor:
    """Return flattened encoder activations for a batch of EEG signals.

    Parameters
    ----------
    signals : Tensor
        Shape ``(B, T, C)`` – the same layout produced by the preprocessing
        pipeline.  If the input is 2-D (single example), a batch dimension is
        added automatically.
    device : str | torch.device, optional
        Target device for the model.  Defaults to *cuda → mps → cpu*.
    """

    global _MODEL

    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else (
                torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            )
        )
    elif isinstance(device, str):
        device = torch.device(device)

    if _MODEL is None:
        _MODEL = _load_encoder(device)

    if signals.dim() == 2:  # (T, C) → (1, T, C)
        signals = signals.unsqueeze(0)

    signals = signals.to(device)

    with torch.no_grad():
        latent = _MODEL(signals.transpose(-1, -2))  # encoder expects (B, C, T)

    # Flatten across channels × features → (B, n_features)
    latent_flat = latent.reshape(latent.shape[0], -1).cpu()

    return latent_flat
