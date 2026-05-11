"""EEGNet-v4 Autoencoder: model definition and inference utilities.

The encoder follows the standard EEGNet-v4 architecture (temporal conv ->
depthwise spatial conv -> separable temporal conv -> pooling) and maps to a
compact latent vector.  The decoder mirrors the encoder to reconstruct
the raw EEG segment.

Public API:
    * ``get_eegnet_ae_model``  - Load a trained checkpoint.
    * ``extract_eegnet_ae``    - Encode a Raw recording into a latent vector.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import mne
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.util import preprocess_time_domain_input

logger = logging.getLogger(__name__)

DEFAULT_N_CHANNELS = 19
DEFAULT_FIXED_LEN = 10 * 128
DEFAULT_LATENT_DIM = 128


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class SeparableConv2d(nn.Module):
    """Depthwise temporal conv followed by pointwise 1x1 conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_len: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=(1, kernel_len),
            padding="same", groups=in_ch, bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class EEGNetAE(nn.Module):
    """EEGNet-v4 autoencoder for time-domain EEG.

    Args:
        n_channels: Number of EEG channels.
        latent_dim: Bottleneck dimensionality.
        fixed_len: Expected input time length (samples).
        F1: Number of temporal filters (block 1).
        D: Depth multiplier for spatial conv.
        F2: Number of filters after separable conv.
        kernel_length: Temporal filter length.
        sep_kernel_len: Separable conv temporal filter length.
        pool1: First pooling factor.
        pool2: Second pooling factor.
        drop_prob: Dropout probability.
    """

    def __init__(
        self,
        n_channels: int = DEFAULT_N_CHANNELS,
        latent_dim: int = DEFAULT_LATENT_DIM,
        fixed_len: int = DEFAULT_FIXED_LEN,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        sep_kernel_len: int = 16,
        pool1: int = 4,
        pool2: int = 8,
        drop_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.fixed_len = fixed_len

        # Encoder
        self.conv_time = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_spat = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(drop_prob)
        self.sepconv = SeparableConv2d(F1 * D, F2, sep_kernel_len)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(drop_prob)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, fixed_len)
            h = self._enc_feats(dummy)
            self.enc_shape = h.shape[1:]
            self.flat_dim = int(h.numel() // h.shape[0])

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # Decoder (mirror)
        self.up2 = nn.Upsample(scale_factor=(1, pool2), mode="nearest")
        self.tsep_T = nn.Conv2d(F2, F2, (1, sep_kernel_len), padding="same", groups=F2, bias=False)
        self.tsep_1x1 = nn.Conv2d(F2, F1 * D, kernel_size=1, bias=False)
        self.dbn3 = nn.BatchNorm2d(F1 * D)
        self.up1 = nn.Upsample(scale_factor=(1, pool1), mode="nearest")
        self.tspat = nn.ConvTranspose2d(F1 * D, F1, (n_channels, 1), groups=F1, bias=False)
        self.dbn2 = nn.BatchNorm2d(F1)
        self.tconv_time = nn.Conv2d(F1, 1, (1, kernel_length), padding="same", bias=False)
        self.out_act = nn.Tanh()

    def _to_4d(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        if x.dim() == 4:
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

    def encode(self, x: Tensor) -> Tensor:
        """Encode EEG to latent vector.

        Args:
            x: Input tensor ``(B, C, T)`` or ``(B, 1, C, T)``.

        Returns:
            Latent tensor ``(B, latent_dim)``.
        """
        x4 = self._to_4d(x)
        h = self._enc_feats(x4)
        return self.fc_mu(h.view(h.size(0), -1))

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to EEG.

        Args:
            z: Latent tensor ``(B, latent_dim)``.

        Returns:
            Reconstructed EEG ``(B, C, T)``.
        """
        h = self.fc_dec(z).view(z.size(0), *self.enc_shape)
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
        if xhat.shape[-1] != self.fixed_len:
            xhat = F.interpolate(xhat, size=self.fixed_len, mode="nearest")
        return xhat.squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        """Full autoencoder pass."""
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def _resolve_latest_ckpt(dataset_name: str = "tuh") -> Path:
    """Find the best EEGNet-AE checkpoint.

    Checks ``EEGNET_AE_CKPT`` env var first, then looks in ``models/``.
    """
    env_path = os.environ.get("EEGNET_AE_CKPT", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    models_dir = _models_dir()
    candidates = list(models_dir.glob(f"{dataset_name}_best.pth"))
    if not candidates:
        candidates = list(models_dir.glob("best.pth"))
        if not candidates:
            return models_dir / "best.pth"
        logger.warning(
            "No dataset-specific (%s) model found, falling back to: %s",
            dataset_name, candidates[0],
        )

    candidates.sort(
        key=lambda p: (p.stat().st_mtime, 1 if "best" in p.name.lower() else 0),
        reverse=True,
    )
    return candidates[0]


def get_eegnet_ae_model(
    *,
    device: Optional[Union[torch.device, str]] = None,
    latent_dim: int = DEFAULT_LATENT_DIM,
    dataset_name: str = "tuh",
) -> EEGNetAE:
    """Load a trained EEGNet-AE model.

    Args:
        device: Torch device (auto-detected if ``None``).
        latent_dim: Latent dimensionality.
        dataset_name: Dataset prefix for checkpoint resolution.

    Returns:
        EEGNetAE model in eval mode.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)

    model = EEGNetAE(
        n_channels=DEFAULT_N_CHANNELS,
        latent_dim=latent_dim,
        fixed_len=DEFAULT_FIXED_LEN,
    )
    ckpt = _resolve_latest_ckpt(dataset_name=dataset_name)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"EEGNet-AE checkpoint not found at {ckpt}. Train the model first."
        )

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_eegnet_ae(
    raw: mne.io.BaseRaw,
    *,
    device: Optional[Union[torch.device, str]] = None,
    latent_dim: int = DEFAULT_LATENT_DIM,
    model: Optional[EEGNetAE] = None,
    dataset_name: str = "tuh",
) -> torch.Tensor:
    """Encode an MNE Raw recording into a 1-D latent vector.

    Args:
        raw: MNE Raw object (pre-cleaned, 19-channel).
        device: Torch device.
        latent_dim: Latent dimensionality.
        model: Pre-loaded model (loaded from checkpoint if ``None``).
        dataset_name: Dataset prefix for checkpoint resolution.

    Returns:
        CPU tensor of shape ``(latent_dim,)``.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    device = torch.device(device)

    if model is None:
        model = get_eegnet_ae_model(device=device, latent_dim=latent_dim, dataset_name=dataset_name)
    data = preprocess_time_domain_input(raw, target_sfreq=128.0, segment_len_sec=10)
    x = torch.as_tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
    z = model.encode(x).squeeze(0).detach().cpu()
    return z


__all__ = ["EEGNetAE", "get_eegnet_ae_model", "extract_eegnet_ae"]
