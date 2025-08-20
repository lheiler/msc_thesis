from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from scipy.signal import welch

from mne.io import read_raw_brainvision
import mne
import numpy as np
import importlib.util
import sys
import matplotlib.pyplot as plt
from utils.util import PSD_CALCULATION_PARAMS

def _load_infer_module():
    this_dir = Path(__file__).resolve().parent
    infer_path = this_dir / "infer.py"
    spec = importlib.util.spec_from_file_location("eegnet_ae_infer", str(infer_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

_infer_mod = _load_infer_module()
EEGNetAE = _infer_mod.EEGNetAE



class TUHFIF60sDataset(torch.utils.data.Dataset):
    """
    Recursively loads .fif EEG files under a root directory (e.g., TUH train/)
    and returns z‑scored segments shaped (C, T) after standard preprocessing.
    Preprocessing: drop A1/A2, pick 19 EEG channels when present, bandpass 3–45 Hz,
    resample to 128 Hz, crop/pad 60 s, z‑score.
    """

    EEG_CHANNELS_19 = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Cz", "Pz", "Fz",
    ]

    def __init__(self, root: Path, segment_len_sec: int = 60, target_sfreq: float = 128.0):
        super().__init__()
        self.root = Path(root)
        self.seg_len = segment_len_sec
        self.sfreq = target_sfreq
        self.files = sorted(self.root.rglob("*.fif"))
        if not self.files:
            raise RuntimeError(f"No .fif files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)

        # Drop A1/A2 if present
        for ch in ("A1", "A2"):
            if ch in raw.ch_names:
                raw.drop_channels([ch])

        # Pick intersection with 19-channel set
        picks_19 = [ch for ch in self.EEG_CHANNELS_19 if ch in raw.ch_names]
        if picks_19:
            raw.pick_channels(picks_19)

        # Crop first (works without preload) so we only load 60s
        sf = raw.info["sfreq"]
        #print the length of the recording in seconds
        #print(f"Recording {path} length: {raw.n_times / sf} seconds")
        raw.crop(tmin=0.0, tmax=self.seg_len - 1.0 / sf)

        # Now load data into memory before filtering/resampling
        raw.load_data(verbose=False)

        # Bandpass and resample (requires preload)
        raw.filter(3.0, 45.0, fir_design="firwin", verbose=False)
        if abs(raw.info["sfreq"] - self.sfreq) > 1e-3:
            raw.resample(self.sfreq, npad="auto")

        data = raw.get_data().astype(np.float32)
        tgt_len = int(self.seg_len * self.sfreq)
        if data.shape[1] < tgt_len:
            pad = tgt_len - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
        elif data.shape[1] > tgt_len:
            data = data[:, :tgt_len]

        data = (data - data.mean()) / (data.std() + 1e-8)
        return torch.from_numpy(data)


# add this helper above mixed_recon_loss
def welch_psd_torch(x: torch.Tensor, fs: float = 128.0, nperseg: int = 256, noverlap: int = 128, eps: float = 1e-12):
    # x: (B, C, T) → returns F: (freqs,), Pxx: (B, C, F)
    B, C, T = x.shape
    x = x.reshape(B * C, T)
    hop = nperseg - noverlap
    window = torch.hann_window(nperseg, device=x.device, dtype=x.dtype)
    S = torch.stft(
        x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
        window=window, center=False, return_complex=True
    )  # (B*C, F, frames)
    Pxx = (S.abs() ** 2).mean(dim=-1)  # Welch average over frames
    scale = fs * (window.pow(2).sum() + eps)  # density-like scaling
    Pxx = Pxx / scale
    F = torch.linspace(0, fs / 2, Pxx.shape[1], device=x.device, dtype=x.dtype)
    return F, Pxx.view(B, C, -1)

# replace your mixed_recon_loss with this
def mixed_recon_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiable spectral loss using Welch PSD; normalized; MSE weight 0.0."""
    # Align time
    Ty, Tt = x_hat.shape[-1], x.shape[-1]
    if Ty < Tt:
        x_hat = nn.functional.pad(x_hat, (0, Tt - Ty))
    elif Ty > Tt:
        x_hat = x_hat[..., :Tt]

    # Welch PSD
    F, Xh_psd = welch_psd_torch(x_hat, fs=128.0, nperseg=256, noverlap=128)
    _, X_psd  = welch_psd_torch(x,     fs=128.0, nperseg=256, noverlap=128)

    # Restrict comparison to configured frequency band (e.g., ≤45 Hz)
    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", 64.0))
    mask = (F >= fmin) & (F <= fmax)
    Xh_psd = Xh_psd[..., mask]
    X_psd  = X_psd[..., mask]

    # Normalize each PSD so that the sum across frequencies is 1 for each (B, C)
    Xh_psd = Xh_psd / (Xh_psd.sum(dim=-1, keepdim=True) + 1e-8)
    X_psd  = X_psd  / (X_psd.sum(dim=-1, keepdim=True)  + 1e-8)

    # Mean normalization (subtract mean and divide by std) for each (B, C)
    Xh_psd = (Xh_psd - Xh_psd.mean(dim=-1, keepdim=True)) / (Xh_psd.std(dim=-1, keepdim=True) + 1e-8)
    X_psd = (X_psd - X_psd.mean(dim=-1, keepdim=True)) / (X_psd.std(dim=-1, keepdim=True) + 1e-8)

    # Spectral MSE in log space
    spec = nn.functional.mse_loss(Xh_psd, X_psd)

    # Time-domain term is disabled
    mse = nn.functional.mse_loss(x_hat, x)
    return 0.0 * mse + 1.0 * spec

def train(
    data_root: Path,
    out_dir: Path,
    *,
    latent_dim: int = 128,
    batch_size: int = 16,
    lr: float = 5e-4,
    epochs: int = 40,
    val_ratio: float = 0.2,
    num_workers: int = 4,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    root = Path(data_root)
    # Auto-detect dataset type: prefer .fif when present
    dataset = TUHFIF60sDataset(root)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = EEGNetAE(n_channels=19, latent_dim=latent_dim, fixed_len=60 * 128).to(device)
    torch.manual_seed(42)
    np.random.seed(42)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)

    best_val = float("inf")
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best.pth"

    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch.to(device).float()  # (B, C, T)
            opt.zero_grad()
            y = model(x)
            loss = mixed_recon_loss(y, x)
            loss.backward()
            opt.step()
            total += loss.item()
        avg_train = total / max(1, len(train_loader))

        model.eval()
        total_v = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device).float()  # (B, C, T)
                y = model(x)
                total_v += mixed_recon_loss(y, x).item()
        avg_val = total_v / max(1, len(val_loader))
        scheduler.step(avg_val)

        print(f"Epoch {epoch:03d} | train {avg_train:.4f} | val {avg_val:.4f}")
        if avg_val < best_val - 1e-4:
            best_val = avg_val
            torch.save({"model_state": model.state_dict()}, best_path)
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print("Early stopping: no val improvement.")
                break
            
        #every 5 epochs save a plot of the psd of the original and reconstructed data
        if epoch % 5 == 0:
            # Compute mean PSD across batch and channels for plotting
            x_np = x.detach().cpu().numpy()  # (B, C, T)
            y_np = y.detach().cpu().numpy()  # (B, C, T)
            window_np = np.hanning(256)
            f, psd_x = welch(x_np, fs=128, nperseg=256, noverlap=128, window=window_np, axis=-1)
            _, psd_y = welch(y_np, fs=128, nperseg=256, noverlap=128, window=window_np, axis=-1)
            psd_x_mean = psd_x.mean(axis=(0, 1))  # (F,)
            psd_y_mean = psd_y.mean(axis=(0, 1))  # (F,)
            plt.figure(figsize=(10, 5))
            plt.plot(f, psd_x_mean)
            plt.plot(f, psd_y_mean)
            plt.legend(["Original", "Reconstructed"])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"PSD of Original and Reconstructed Data at Epoch {epoch}")
            plt.savefig(f"plots/psd_plot_{epoch}.png")
            plt.close()

    print(f"Saved best model to {best_path}")
    return best_path


def main():
    p = argparse.ArgumentParser()
    data_root = Path("/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/train")
    out_dir = Path("/homes/lrh24/thesis/code/latent_extraction/EEGNet-AE/")
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    train(
        data_root=Path(data_root),
        out_dir=Path(out_dir),
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
