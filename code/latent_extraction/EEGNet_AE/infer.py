from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import mne
import numpy as np
import torch
from torch import nn, Tensor
from utils.util import preprocess_time_domain_input
import os
import torch.nn.functional as F


# ----------------- Helper: Depthwise-separable conv -----------------
class SeparableConv2d(nn.Module):
    """Depthwise temporal conv followed by pointwise 1x1 conv."""
    def __init__(self, in_ch: int, out_ch: int, kernel_len: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=(1, kernel_len),
            padding='same', groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))



class EEGNetAE(nn.Module):
    """
    EEGNet-v4 autoencoder. Encoder matches EEGNet v4 blocks.
    Decoder mirrors the blocks to reconstruct raw EEG.

    Input to encode/forward: (B, C, T) with C == n_channels.
    Output of forward: (B, C, T).
    """

    def __init__(
        self,
        n_channels: int = 19,
        latent_dim: int = 128,
        fixed_len: int = 60 * 128,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        sep_kernel_len: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        drop_prob: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.fixed_len = fixed_len

        # ---------------- Encoder (EEGNet v4) ----------------
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding='same',
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv across electrodes
        self.conv_spat = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(drop_prob)

        # Depthwise-separable temporal conv
        self.sepconv = SeparableConv2d(F1 * D, F2, sep_kernel_len)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(drop_prob)

        # Infer encoder feature map shape and flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, fixed_len)
            h = self._enc_feats(dummy)
            self.enc_shape = h.shape[1:]          # (F2, 1, T2)
            self.flat_dim = int(h.numel() // h.shape[0])

        # Latent mapping
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # ---------------- Decoder (mirror) ----------------
        # Invert pooling by upsampling, then invert separable and spatial/temporal convs
        self.up2 = nn.Upsample(scale_factor=(1, pool2), mode='nearest')
        # Use standard convs with 'same' padding to preserve time length after upsampling
        self.tsep_T = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, sep_kernel_len),
            padding='same',
            groups=F2,
            bias=False,
        )
        self.tsep_1x1 = nn.Conv2d(F2, F1 * D, kernel_size=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(F1 * D)

        self.up1 = nn.Upsample(scale_factor=(1, pool1), mode='nearest')
        self.tspat = nn.ConvTranspose2d(
            in_channels=F1 * D,
            out_channels=F1,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.dbn2 = nn.BatchNorm2d(F1)

        self.tconv_time = nn.Conv2d(
            in_channels=F1,
            out_channels=1,
            kernel_size=(1, kernel_length),
            padding='same',
            bias=False,
        )
        self.out_act = nn.Tanh()

    # ---- internals ----
    def _to_4d(self, x: Tensor) -> Tensor:
        # Expect (B, C, T). Convert to (B, 1, C, T)
        if x.dim() == 3:
            return x.unsqueeze(1)
        elif x.dim() == 4:
            return x
        raise ValueError(f"Expected (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")

    def _enc_feats(self, x4: Tensor) -> Tensor:
        x = self.conv_time(x4)
        x = self.bn1(x)
        x = self.conv_spat(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sepconv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    # ---- public API ----
    def encode(self, x: Tensor) -> Tensor:
        x4 = self._to_4d(x)
        h = self._enc_feats(x4)
        z = self.fc_mu(h.view(h.size(0), -1))
        return z

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_dec(z).view(z.size(0), *self.enc_shape)  # (B, F2, 1, T2)
        h = self.up2(h)
        h = self.tsep_T(h)
        h = self.tsep_1x1(h)
        h = self.dbn3(h)
        h = F.elu(h)
        h = self.up1(h)
        h = self.tspat(h)
        h = self.dbn2(h)
        h = F.elu(h)
        xhat = self.tconv_time(h)
        xhat = self.out_act(xhat)
        # Enforce exact time length equality
        if xhat.shape[-1] != self.fixed_len:
            xhat = F.interpolate(xhat, size=self.fixed_len, mode='nearest')
        return xhat.squeeze(1)  # (B, C, T)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        return self.decode(z)


# -----------------------------------------------------------------------------
#                           INFERENCE UTILITIES
# -----------------------------------------------------------------------------

_EEG_CHANNELS_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Cz", "Pz", "Fz",
]


def _preprocess_raw(raw: mne.io.BaseRaw, *, target_sfreq: float = 128.0, segment_len_sec: int = 60) -> np.ndarray:
    """Preprocess Raw to (C, T) without redundant channel cleaning (handled in extractor)."""
    return preprocess_time_domain_input(raw, target_sfreq=target_sfreq, segment_len_sec=segment_len_sec)


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _models_dir() -> Path:
    return _this_dir() / "models"


def _resolve_latest_ckpt() -> Path:
    # Allow override via env var
    env_path = os.environ.get("EEGNET_AE_CKPT", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    # Prefer newest checkpoint within models/ (favor files containing 'best')
    models_dir = _models_dir()
    candidates = list(models_dir.glob("*.pth"))
    if not candidates:
        # Backward-compatible default
        return models_dir / "best.pth"
    def sort_key(p: Path):
        return (p.stat().st_mtime, 1 if "best" in p.name.lower() else 0)
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def get_eegnet_ae_model(*, device: Optional[torch.device | str] = None, latent_dim: int = 128) -> EEGNetAE:
    """Load the EEGNetAE model with the best checkpoint if available."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device)

    model = EEGNetAE(n_channels=19, latent_dim=latent_dim, fixed_len=60 * 128)
    ckpt = _resolve_latest_ckpt()
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])  # wrapped format
        else:
            model.load_state_dict(state)  # raw state_dict
    else:
        raise FileNotFoundError(
            f"EEGNet-AE checkpoint not found at {ckpt}. Train the model first using train.py."
        )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_eegnet_ae(
    raw: mne.io.BaseRaw,
    *,
    device: Optional[torch.device | str] = None,
    latent_dim: int = 128,
    model: Optional[EEGNetAE] = None,
) -> torch.Tensor:
    """Return a 1‑D latent vector for the given Raw recording.

    The vector is returned on CPU for safe serialisation. Shape: (latent_dim,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device)

    if model is None:
        model = get_eegnet_ae_model(device=device, latent_dim=latent_dim)
    data = _preprocess_raw(raw)
    x = torch.as_tensor(data, dtype=torch.float32, device=device).unsqueeze(0)  # (1, C, T)
    z = model.encode(x).squeeze(0).detach().cpu()  # (latent_dim,)
    return z


__all__ = ["EEGNetAE", "get_eegnet_ae_model", "extract_eegnet_ae"]
