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
# NOTE: _THIS_DIR = <project_root>/code/latent_extraction
# We only need to go up one level to reach the project root "code/".
# Using parents[1] ensures the path resolves to the correct directory even if
# the project is nested more deeply.
_PROJECT_ROOT = _THIS_DIR.parents[1]  # /…/code/

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
    """Return the path to *newoptuna_best.pth* (preferred) or *optuna_best.pth* or *best.pth*.

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

    # Search depth-first for newoptuna_best.pth → optuna_best.pth → best.pth
    patterns: List[Tuple[str, str]] = [
        ("newoptuna_best.pth", "**/newoptuna_best.pth"),
        ("optuna_best.pth", "**/optuna_best.pth"),
        ("best.pth", "**/best.pth"),
        ("*.pth", "**/*.pth"),
    ]

    for _name, glob_pat in patterns:
        matches = list(weights_root.glob(glob_pat))
        if matches:
            # Return the *latest* file if multiple exist (sorted by mtime)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
    raise FileNotFoundError(
        "No suitable CwA-T weight file found (newoptuna_best.pth / optuna_best.pth / best.pth)."
    )


def _load_best_params() -> dict:
    """Load the hyperparameters from the best trial JSON file."""
    weights_root = _PROJECT_ROOT / "extras" / "models" / "weights"
    
    # Look for the params file in the same directory as the weights
    for pattern in ["**/newoptuna_best_params.json", "**/optuna_best_params.json"]:
        matches = list(weights_root.glob(pattern))
        if matches:
            # Use the latest file
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            params_path = matches[0]
            import json
            with params_path.open("r") as f:
                return json.load(f)
    
    # Fallback: return empty dict if no params file found
    return {}


def _load_encoder(device: torch.device) -> nn.Module:
    """Instantiate the encoder and load the saved weights.

    Uses the exact hyperparameters from the saved JSON file if available,
    otherwise falls back to guessing from the state dict shape.
    """

    state_dict_path = _find_best_weights()
    print("STATE DICT PATH")
    print(state_dict_path)
    raw_state = torch.load(state_dict_path, map_location="cpu")

    # ------------------------------------------------------------------
    # The checkpoint might contain the *full* CwA-T model (encoder +
    # transformer head).  In that case, parameter keys are prefixed with
    # "encoder." (or sometimes "module.encoder." when saved via
    # `torch.nn.DataParallel`).  We strip the leading prefix so that the
    # state-dict is compatible with the *stand-alone* encoder instance
    # created here.
    # ------------------------------------------------------------------
    def _extract_encoder_subdict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if any(k.startswith("encoder.") for k in sd):
            return {k.split("encoder.", 1)[1]: v for k, v in sd.items() if k.startswith("encoder.")}
        if any(k.startswith("module.encoder.") for k in sd):
            return {k.split("module.encoder.", 1)[1]: v for k, v in sd.items() if k.startswith("module.encoder.")}
        # Already a pure encoder state-dict
        return sd

    # Some checkpoints store the dict under a wrapper key (e.g. "state_dict")
    state_dict = (
        _extract_encoder_subdict(raw_state["state_dict"]) if isinstance(raw_state, dict) and "state_dict" in raw_state else _extract_encoder_subdict(raw_state)
    )

    # Try to load exact hyperparameters from JSON file
    best_params = _load_best_params()
    
    if best_params and "d_model" in best_params:
        # Use exact parameters from the best trial
        d_model = best_params["d_model"]
        n_channels = 19  # fixed for TUH / Harvard datasets
        
        # Determine encoder variant from the model name in the path
        model_name = state_dict_path.parent.name
        if "encoderL" in model_name.upper():
            enc_fn = res_encoderL
            variant = "L"
        elif "encoderS" in model_name.upper():
            enc_fn = res_encoderS
            variant = "S"
        else:
            enc_fn = res_encoderM  # default
            variant = "M"
        
        try:
            encoder = enc_fn(
                n_channels=n_channels,
                groups=n_channels,
                num_classes=2,  # unused; kept for signature compatibility
                d_model=d_model,
            )
            encoder.load_state_dict(state_dict, strict=True)
            encoder.eval()
            encoder.to(device)
            
            print(
                f"[CwA-T] Loaded encoder variant={variant} d_model={d_model} "
                f"from {state_dict_path.relative_to(_PROJECT_ROOT)} "
                f"(using exact params from JSON)"
            )
            return encoder
        except (RuntimeError, KeyError) as e:
            print(f"Failed to load with exact params: {e}")
            print("Falling back to parameter guessing...")
    
    # Fallback: guess parameters from state dict (original logic)
    # Expanded set of plausible d_model sizes – covers typical multiples of 64
    # as well as the common 512/768 dimensions used in transformer setups.
    candidate_d_models = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 448, 512, 640, 768]
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
                    f"from {state_dict_path.relative_to(_PROJECT_ROOT)} "
                    f"(guessed parameters)"
                )

                return encoder
            except (RuntimeError, KeyError):
                # Shape mismatch → try next candidate
                continue

    raise RuntimeError(
        "Could not match the saved weights with any known encoder variant. "
        f"Weights path: {state_dict_path}"
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

    # ------------------------------------------------------------------
    # Accept MNE Raw objects directly – convert to torch tensor
    # ------------------------------------------------------------------
    if not torch.is_tensor(signals):
        try:
            import mne  # local import to avoid hard dependency unless needed
            if isinstance(signals, mne.io.BaseRaw):
                raw = signals.copy()
                TARGET_SFREQ = 100  # Hz – matches training config
                TARGET_LEN = 6000   # samples (60 s at 100 Hz)

                # Resample if necessary
                if int(raw.info["sfreq"]) != TARGET_SFREQ:
                    raw = raw.resample(TARGET_SFREQ)

                data = raw.get_data(picks="eeg")  # (C, T)
                data = data[:19, :]  # use first 19 standard channels

                # Crop / pad to fixed length
                if data.shape[1] >= TARGET_LEN:
                    data = data[:, :TARGET_LEN]
                else:
                    import numpy as np
                    pad_width = TARGET_LEN - data.shape[1]
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

                signals = torch.tensor(data.T, dtype=torch.float32)
            else:
                raise TypeError
        except (ImportError, TypeError):
            raise ValueError(
                "extract_cwat received input of unsupported type. Expected a torch.Tensor "
                "or an MNE Raw object with EEG data."
            )

    if signals.dim() == 2:  # (T, C) → (1, T, C)
        signals = signals.unsqueeze(0)

    signals = signals.to(device)

    with torch.no_grad():
        latent = _MODEL(signals.transpose(-1, -2))  # encoder expects (B, C, T)

    # Flatten across channels × features → (B, n_features)
    latent_flat = latent.reshape(latent.shape[0], -1).cpu()

    return latent_flat
