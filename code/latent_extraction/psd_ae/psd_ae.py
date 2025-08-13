#create super simple psd auto encoder that gets PSDs from the raw data with the standard preprocessing

import torch
import torch.nn as nn
import numpy as np
import mne
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
import sys
import random

# Add the utils directory to the Python path
utils_path = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from gen_dataset import TUHFIF60sDataset

SEED = 42

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compute_welch_psd(x: torch.Tensor, sfreq: float, *, n_fft: int = 256,
                        n_per_seg: int = 256, n_overlap: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Welch PSD per channel for a batch of segments.

    Args:
        x: Tensor shaped (B, C, T) or (C, T), values in time domain.
        sfreq: Sampling frequency in Hz.
        n_fft: FFT length.
        n_per_seg: Segment length for Welch.
        n_overlap: Overlap between segments for Welch.

    Returns:
        psd: Tensor shaped (B, C, F) with power spectral density.
        freqs: Tensor shaped (F,) with frequency bins.
    """
    x_np = x.detach().cpu().numpy()
    if x_np.ndim == 2:  # (C, T) -> (1, C, T)
        x_np = x_np[None, ...]
    # mne returns (psds, freqs)
    psds, freqs = mne.time_frequency.psd_array_welch(
        x_np,
        sfreq=sfreq,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    psd_t = torch.from_numpy(psds.astype(np.float32))
    f_t = torch.from_numpy(freqs.astype(np.float32))
    return psd_t, f_t


class PSDAE(nn.Module):
    """Simple autoencoder that operates on Power Spectral Density (PSD) features."""
    
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        
        assert input_dim // 4 >= latent_dim, (
            f"latent_dim={latent_dim} must be <= input_dim//4={input_dim // 4} for the current architecture"
        )
        
        # Encoder: PSD -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, latent_dim)
        )
        
        # Decoder: latent -> PSD
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode PSD to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to PSD."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        z = self.encode(x)
        return self.decode(z)


def train(model, train_loader, val_loader, device, sfreq: float, epochs: int = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            # batch: (B, C, T) time-domain
            psd, _ = _compute_welch_psd(batch, sfreq)
            # treat each channel PSD as a separate input vector
            B, C, F = psd.shape
            inputs = psd.reshape(B * C, F).to(device)
            # Per-vector normalization: zero mean, unit std
            mu = inputs.mean(dim=1, keepdim=True)
            std = inputs.std(dim=1, keepdim=True).clamp_min(1e-8)
            inputs = (inputs - mu) / std

            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1
        print(f"Epoch {epoch} loss: {total_loss / max(1, n_steps):.6f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_steps = 0
            for batch in val_loader:
                psd, _ = _compute_welch_psd(batch, sfreq)
                B, C, F = psd.shape
                inputs = psd.reshape(B * C, F).to(device)
                mu = inputs.mean(dim=1, keepdim=True)
                std = inputs.std(dim=1, keepdim=True).clamp_min(1e-8)
                inputs = (inputs - mu) / std
                recon = model(inputs)
                val_total += float(criterion(recon, inputs).item())
                val_steps += 1
            print(f"Epoch {epoch} val loss: {val_total / max(1, val_steps):.6f}")
        model.train()
        

if __name__ == "__main__":
    set_seed()
    device = get_device()
    # Load dataset (time-domain segments)
    latent_dim = 8
    
    dataset = TUHFIF60sDataset(root="/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/train/")
    print(f"Loaded {len(dataset)} files")

    # Infer PSD input dimension from a single example
    with torch.no_grad():
        sample = dataset[0].unsqueeze(0)  # (1, C, T)
        psd_sample, freqs = _compute_welch_psd(sample, sfreq=dataset.sfreq)
        input_dim = int(psd_sample.shape[-1])

    # Create model for per-channel PSD vectors
    model = PSDAE(input_dim=input_dim, latent_dim=latent_dim).to(device)

    from torch.utils.data import random_split
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    train_ds, val_ds = random_split(
        dataset, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    train(model, train_loader, val_loader, device=device, sfreq=dataset.sfreq)

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "freqs": freqs.numpy(),
        "latent_dim": latent_dim,
        "input_dim": input_dim,
    }, f"models/psd_ae_{latent_dim}.pth")