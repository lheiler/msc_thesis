import os
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import torch
from utils.util import preprocess_time_domain_input

# Reuse the EEG2Rep project modules
EEG2REP_ROOT = Path(__file__).resolve().parent / "EEG2Rep"


def _load_project_modules():
    import sys
    root_str = str(EEG2REP_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from Models.model import Encoder_factory  # type: ignore
    from Models.utils import load_model  # type: ignore
    return Encoder_factory, load_model
def _resolve_latest_eeg2rep_checkpoint() -> Path:
    """Return the newest checkpoint under EEG2Rep/Results recursively.

    Prefers files named '*best*.pth' over others when timestamps tie.
    """
    results_root = EEG2REP_ROOT / "EEG2Rep" / "Results"
    if not results_root.exists():
        # Fallback to older layout if present
        results_root = EEG2REP_ROOT / "Results"
    candidates = list(results_root.rglob("checkpoints/*.pth")) if results_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No EEG2Rep checkpoints found under {results_root}")
    # Choose most recent by mtime; break ties by favoring '*best*'
    def sort_key(p: Path):
        return (p.stat().st_mtime, 1 if "best" in p.name.lower() else 0)
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]



def _preprocess_raw_to_tensor(
    raw: mne.io.BaseRaw,
    segment_len_sec: int = 10,
    target_sfreq: float = 128.0,
) -> torch.Tensor:
    """
    Convert an MNE Raw into a single (1, C, T) float32 tensor using the
    same pipeline as EEG2Rep training on FIF data.
    """
    # Preprocess time-domain data (cleaning already done in extractor)
    data = preprocess_time_domain_input(raw, target_sfreq=target_sfreq, segment_len_sec=segment_len_sec)

    # Require exactly 19 channels to match model architecture
    if data.shape[0] != 19:
        # If not exactly 19 after picking, refuse (model was trained with 19)
        raise RuntimeError(f"EEG2Rep expects 19 channels, got {data.shape[0]}")

    tensor = torch.from_numpy(data).unsqueeze(0)  # (1, C, T)
    return tensor


def get_eeg2rep_model(
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    segment_len_sec: int = 10,
    target_sfreq: float = 128.0,
) -> torch.nn.Module:
    """
    Build EEG2Rep encoder and load weights from a checkpoint.

    Parameters
    - device: target torch.device
    - checkpoint_path: explicit path to a saved model (.pth). If None, uses
      environment variable EEG2REP_CKPT, else falls back to the known default
      from training results.
    - segment_len_sec, target_sfreq: define the expected (C, T) shape

    Returns
    - model on device, ready for .linear_prob(x)
    """
    Encoder_factory, load_model = _load_project_modules()

    ckpt_env = os.environ.get("EEG2REP_CKPT")
    ckpt_path = Path(checkpoint_path) if checkpoint_path else (Path(ckpt_env) if ckpt_env else _resolve_latest_eeg2rep_checkpoint())
    if not ckpt_path.exists():
        raise FileNotFoundError(f"EEG2Rep checkpoint not found: {ckpt_path}")

    # Derive configuration from the checkpoint directory
    # results_dir/..../<timestamp>/{configuration.json, checkpoints/}
    cfg_path = ckpt_path.parent.parent / "configuration.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"EEG2Rep configuration.json not found next to checkpoint: {cfg_path}")

    import json
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # Minimal config needed to construct the model
    cfg["device"] = device
    cfg["Data_shape"] = (1, 19, int(segment_len_sec * target_sfreq))
    cfg["num_labels"] = 1

    model = Encoder_factory(cfg)
    model.to(device)

    # Load weights (model only)
    model = load_model(model, str(ckpt_path))
    model.eval()
    return model


@torch.no_grad()
def extract_eeg2rep(
    raw: mne.io.BaseRaw,
    device: torch.device,
    model: torch.nn.Module,
    segment_len_sec: int = 10,
    target_sfreq: float = 128.0,
) -> np.ndarray:
    """
    Compute the EEG2Rep latent vector from an MNE Raw.

    Returns a 1D numpy array (embedding).
    """
    x = _preprocess_raw_to_tensor(raw, segment_len_sec=segment_len_sec, target_sfreq=target_sfreq)
    x = x.to(device)
    # Use linear_prob which averages contextual encoder outputs across time
    if not hasattr(model, "linear_prob"):
        raise AttributeError("EEG2Rep model missing linear_prob method")
    vec = model.linear_prob(x)  # (1, D)
    return vec.squeeze(0).detach().cpu().numpy().astype(np.float32)


