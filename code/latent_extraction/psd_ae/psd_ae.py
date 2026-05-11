"""PSD Autoencoder for latent feature extraction.

A simple symmetric autoencoder operating on normalised Welch PSD vectors.
The encoder compresses a frequency-domain representation into a compact
latent code; the decoder reconstructs the original PSD.

Public API for inference:
    * ``get_psd_ae_model``      - Load a trained checkpoint.
    * ``extract_psd_ae_avg``    - Encode the channel-averaged PSD.
    * ``extract_psd_ae_channel``- Encode per-channel PSDs (concatenated).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.util import (
    PSD_CALCULATION_PARAMS,
    compute_psd_from_array,
    compute_psd_from_raw,
    normalize_psd,
)

logger = logging.getLogger(__name__)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> str:
    """Select the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _plot_recon_example(
    inputs: torch.Tensor,
    recon: torch.Tensor,
    freqs: torch.Tensor,
    *,
    path: Path,
) -> None:
    """Save a single input vs. reconstruction plot."""
    x = inputs[0].detach().cpu().float().numpy()
    y = recon[0].detach().cpu().float().numpy()
    f = freqs.detach().cpu().float().numpy()
    fig, ax = plt.subplots()
    ax.plot(f, x, label="input")
    ax.plot(f, y, label="recon")
    ax.set_xlabel("Hz")
    ax.set_ylabel("norm PSD")
    ax.legend(loc="best")
    fig.savefig(str(path))
    plt.close(fig)


class PSDAE(nn.Module):
    """Symmetric autoencoder operating on PSD feature vectors.

    Args:
        input_dim: Number of frequency bins.
        latent_dim: Bottleneck dimensionality.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64) -> None:
        super().__init__()
        assert input_dim // 4 >= latent_dim, (
            f"latent_dim={latent_dim} must be <= input_dim//4={input_dim // 4}"
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode PSD to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to PSD space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass (encode then decode)."""
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_latest_checkpoint(dataset_name: str = "tuh") -> Path:
    """Find the newest PSD-AE checkpoint for a dataset.

    Args:
        dataset_name: Dataset prefix to search for.

    Returns:
        Path to the checkpoint file.

    Raises:
        RuntimeError: If no checkpoint is found.
    """
    models_dir = Path(__file__).resolve().parent / "models"
    candidates = sorted(
        models_dir.glob(f"{dataset_name}_psd_ae_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            models_dir.glob("psd_ae_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"No PSD-AE checkpoint found in {models_dir}. "
                "Train via latent_extraction/psd_ae/psd_ae.py"
            )
        logger.warning(
            "No dataset-specific (%s) model found, falling back to: %s",
            dataset_name, candidates[0],
        )
    else:
        logger.info("Using checkpoint %s", candidates[0])
    return candidates[0]


def get_psd_ae_model(
    device: Union[str, torch.device] = "cpu",
    ckpt_path: Optional[str] = None,
    dataset_name: str = "tuh",
) -> PSDAE:
    """Load a trained PSD-AE model.

    Args:
        device: Torch device.
        ckpt_path: Explicit checkpoint path (auto-resolved if ``None``).
        dataset_name: Dataset prefix for checkpoint resolution.

    Returns:
        PSDAE model in eval mode on the specified device.
    """
    if ckpt_path is None:
        ckpt = _resolve_latest_checkpoint(dataset_name=dataset_name)
    else:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"PSD-AE checkpoint not found: {ckpt}")

    try:
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(ckpt), map_location="cpu")

    if "input_dim" not in payload:
        raise RuntimeError(f"PSD-AE checkpoint missing 'input_dim': {ckpt}")

    input_dim = int(payload["input_dim"])
    latent_dim = int(payload.get("latent_dim", 64))

    model = PSDAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_psd_ae_avg(
    raw: mne.io.BaseRaw,
    *,
    device: Union[str, torch.device] = "cpu",
    model: Optional[PSDAE] = None,
    ckpt_path: Optional[str] = None,
) -> np.ndarray:
    """Encode the channel-averaged PSD into a latent vector.

    Args:
        raw: MNE Raw object.
        device: Torch device.
        model: Pre-loaded model (loaded from checkpoint if ``None``).
        ckpt_path: Explicit checkpoint path.

    Returns:
        Flattened float32 latent array.
    """
    model = model or get_psd_ae_model(device=device, ckpt_path=ckpt_path)
    psd_avg_np = compute_psd_from_raw(raw, calculate_average=True, normalize=True)
    psd_avg = torch.from_numpy(psd_avg_np.astype(np.float32)).unsqueeze(0).to(device)
    z = model.encode(psd_avg)
    return z.detach().cpu().numpy().flatten()


@torch.no_grad()
def extract_psd_ae_channel(
    raw: mne.io.BaseRaw,
    *,
    device: Union[str, torch.device] = "cpu",
    model: Optional[PSDAE] = None,
    ckpt_path: Optional[str] = None,
) -> np.ndarray:
    """Encode per-channel PSDs into a concatenated latent vector.

    Args:
        raw: MNE Raw object.
        device: Torch device.
        model: Pre-loaded model (loaded from checkpoint if ``None``).
        ckpt_path: Explicit checkpoint path.

    Returns:
        Flattened float32 latent array (channels x latent_dim).
    """
    model = model or get_psd_ae_model(device=device, ckpt_path=ckpt_path)
    psd_np = compute_psd_from_raw(raw, calculate_average=False, normalize=True)
    psd_t = torch.from_numpy(psd_np.astype(np.float32)).to(device)
    z = model.encode(psd_t).detach().cpu().numpy()
    return z.flatten()


# ---------------------------------------------------------------------------
# Training (standalone script)
# ---------------------------------------------------------------------------

def train(
    model: PSDAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    sfreq: float,
    *,
    epochs: int = 100,
    patience: int = 5,
    save_prefix: str = "tuh",
    latent_dim: int = 8,
    input_dim: int = 0,
) -> None:
    """Train the PSD-AE with early stopping.

    Args:
        model: PSDAE instance.
        train_loader: Training DataLoader (yields ``(B, C, T)`` tensors).
        val_loader: Validation DataLoader.
        device: Torch device string.
        sfreq: Sampling frequency of the input data.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        save_prefix: Dataset name prefix for checkpoint file.
        latent_dim: Latent dimensionality (for checkpoint metadata).
        input_dim: Input frequency bins (for checkpoint metadata).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    n_fft = int(PSD_CALCULATION_PARAMS["n_fft"])
    n_overlap = int(PSD_CALCULATION_PARAMS["n_overlap"])
    n_per_seg = int(PSD_CALCULATION_PARAMS["n_per_seg"])
    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 1.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", 45.0))
    seg_len = int(PSD_CALCULATION_PARAMS["segment_length"] * sfreq)

    dummy = np.zeros(seg_len, dtype=np.float32)
    _, freqs_np = mne.time_frequency.psd_array_welch(
        dummy[None, :], sfreq=float(sfreq),
        n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
        average="mean", verbose=False, fmin=fmin, fmax=fmax,
    )
    freqs_t = torch.from_numpy(freqs_np.astype(np.float32))

    best_val_loss: Optional[float] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    bad_epochs = 0

    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            B, C, T = batch.shape
            x_np = batch.detach().cpu().numpy().astype(np.float32)
            x2d = x_np.reshape(B * C, T)
            psd_2d, _ = mne.time_frequency.psd_array_welch(
                x2d, sfreq=float(sfreq),
                n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                average="mean", verbose=False, fmin=fmin, fmax=fmax,
            )
            inputs = torch.from_numpy(
                normalize_psd(psd_2d.astype(np.float32))
            ).to(device)

            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1

        avg_train = total_loss / max(1, n_steps)
        logger.info("Epoch %d train loss: %.6f", epoch, avg_train)

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_steps = 0
            for batch in val_loader:
                B, C, T = batch.shape
                x_np = batch.detach().cpu().numpy().astype(np.float32)
                x2d = x_np.reshape(B * C, T)
                psd_2d, _ = mne.time_frequency.psd_array_welch(
                    x2d, sfreq=float(sfreq),
                    n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
                    average="mean", verbose=False, fmin=fmin, fmax=fmax,
                )
                inputs = torch.from_numpy(
                    normalize_psd(psd_2d.astype(np.float32))
                ).to(device)
                recon = model(inputs)
                if val_steps == 1:
                    Path("plots").mkdir(exist_ok=True)
                    _plot_recon_example(
                        inputs.detach().cpu(), recon.detach().cpu(),
                        freqs_t.detach().cpu(),
                        path=Path("plots/val_recon_example.png"),
                    )
                val_total += float(criterion(recon, inputs).item())
                val_steps += 1
            val_loss = val_total / max(1, val_steps)
            logger.info("Epoch %d val loss: %.6f", epoch, val_loss)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                Path("models").mkdir(parents=True, exist_ok=True)
                save_path = Path(f"models/{save_prefix}_psd_ae_{latent_dim}.pth")
                torch.save({
                    "state_dict": model.state_dict(),
                    "freqs": torch.from_numpy(freqs_np.astype(np.float32)),
                    "latent_dim": latent_dim,
                    "input_dim": input_dim,
                }, str(save_path))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val=%.6f)",
                        epoch, best_val_loss,
                    )
                    break

        model.train()
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    root_path = _Path(__file__).resolve().parent.parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from data_preprocessing.gen_dataset import TUHFIF60sDataset
    from torch.utils.data import random_split

    parser = argparse.ArgumentParser(description="Train PSD-AE model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to train_epochs.pkl")
    parser.add_argument("--dataset_name", type=str, default="tuh", help="Dataset name prefix")
    args = parser.parse_args()

    set_seed()
    device = _get_device()
    logger.info("Starting PSD-AE training run")

    _latent_dim = 8
    _batch_size = 64
    dataset = TUHFIF60sDataset(args.data_root)
    logger.info("Loaded %d files", len(dataset))

    seg_len = int(PSD_CALCULATION_PARAMS["segment_length"] * dataset.sfreq)
    dummy = np.zeros(seg_len, dtype=np.float32)
    _, freqs_np = compute_psd_from_array(dummy, sfreq=dataset.sfreq, return_freqs=True, normalize=False)
    _input_dim = int(freqs_np.shape[0])

    _model = PSDAE(input_dim=_input_dim, latent_dim=_latent_dim).to(device)

    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    train_ds, val_ds = random_split(
        dataset, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=_batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=_batch_size, shuffle=False, num_workers=4)

    logger.info("train=%d val=%d", len(train_ds), len(val_ds))

    train(
        _model, train_loader, val_loader, device=device, sfreq=dataset.sfreq,
        save_prefix=args.dataset_name, latent_dim=_latent_dim, input_dim=_input_dim,
    )
    logger.info("Training finished")
