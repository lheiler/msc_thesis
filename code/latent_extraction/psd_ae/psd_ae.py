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
from typing import Union
import wandb
import matplotlib.pyplot as plt

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

def _wandb_log_recon_example(inputs: torch.Tensor, recon: torch.Tensor, freqs: torch.Tensor, *, tag: str, step: int) -> None:
    """Log a single PSD reconstruction plot to W&B.

    Args:
        inputs: Tensor (N, F) normalized PSD inputs.
        recon: Tensor (N, F) reconstructed PSDs.
        freqs: Tensor (F,) frequency bins in Hz.
        tag: W&B image tag.
        step: global step or epoch for logging.
    """
    try:
        idx = 0
        x = inputs[idx].detach().cpu().float().numpy()
        y = recon[idx].detach().cpu().float().numpy()
        f = freqs.detach().cpu().float().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(f, x, label="input")
        ax.plot(f, y, label="recon")
        ax.set_xlabel("Hz")
        ax.set_ylabel("norm PSD")
        ax.legend(loc="best")
        wandb.log({tag: wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        # do not raise during training; logging is best-effort
        print(f"W&B recon log failed: {e}")

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


def _resolve_latest_checkpoint() -> Path:
    """Find the newest PSD-AE checkpoint in the models directory.

    Returns:
        Path: Path to the newest checkpoint file.
    Raises:
        RuntimeError: If no checkpoint is found.
    """
    models_dir = Path(__file__).resolve().parent / "models"
    candidates = sorted(models_dir.glob("psd_ae_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No PSD-AE checkpoint found in {models_dir}. Train via latent_extraction/psd_ae/psd_ae.py")
    return candidates[0]


def get_psd_ae_model(device: Union[str, torch.device] = "cpu", ckpt_path: Optional[str] = None) -> PSDAE:
    """Load PSD-AE model from a checkpoint and put it on device.

    Args:
        device: Torch device or string.
        ckpt_path: Optional explicit checkpoint path. If None, use latest in models/.

    Returns:
        PSDAE: Model loaded and set to eval mode on the specified device.
    """
    if ckpt_path is None:
        ckpt = _resolve_latest_checkpoint()
    else:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"PSD-AE checkpoint not found: {ckpt}")

    payload = torch.load(str(ckpt), map_location="cpu")
    if "input_dim" not in payload:
        raise RuntimeError(f"PSD-AE checkpoint missing 'input_dim': {ckpt}")
    input_dim = int(payload["input_dim"])  # frequency bins
    latent_dim = int(payload.get("latent_dim", 64))

    model = PSDAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_psd_ae_avg(raw: mne.io.BaseRaw, *, device: Union[str, torch.device] = "cpu",
                       model: Optional[PSDAE] = None, ckpt_path: Optional[str] = None) -> torch.Tensor:
    """Extract a single latent vector by encoding the channel-averaged PSD.

    Normalization matches training: per-vector zero mean and unit std before encode.
    """
    model = model or get_psd_ae_model(device=device, ckpt_path=ckpt_path)
    sfreq = float(raw.info.get("sfreq", 128.0))

    data_np = raw.get_data().astype(np.float32)  # (C, T)
    data_t = torch.from_numpy(data_np)
    psd_t, _ = _compute_welch_psd(data_t, sfreq, n_fft=256, n_per_seg=256, n_overlap=128)
    psd_t = psd_t.squeeze(0)  # (C, F)

    psd_avg = psd_t.mean(dim=0, keepdim=True).to(device)  # (1, F)
    mu = psd_avg.mean(dim=1, keepdim=True)
    std = psd_avg.std(dim=1, keepdim=True).clamp_min(1e-8)
    psd_norm = (psd_avg - mu) / std

    z = model.encode(psd_norm).squeeze(0)
    return z.detach().cpu()


@torch.no_grad()
def extract_psd_ae_channel(raw: mne.io.BaseRaw, *, device: Union[str, torch.device] = "cpu",
                           model: Optional[PSDAE] = None, ckpt_path: Optional[str] = None) -> torch.Tensor:
    """Extract a single latent vector by encoding each channel PSD then averaging latents.

    Each channel vector is normalized independently before encoding to match training.
    """
    model = model or get_psd_ae_model(device=device, ckpt_path=ckpt_path)
    sfreq = float(raw.info.get("sfreq", 128.0))

    data_np = raw.get_data().astype(np.float32)  # (C, T)
    data_t = torch.from_numpy(data_np)
    psd_t, _ = _compute_welch_psd(data_t, sfreq, n_fft=256, n_per_seg=256, n_overlap=128)
    psd_t = psd_t.squeeze(0)  # (C, F)

    # Per-channel normalization
    mu = psd_t.mean(dim=1, keepdim=True)
    std = psd_t.std(dim=1, keepdim=True).clamp_min(1e-8)
    psd_norm = ((psd_t - mu) / std).to(device)  # (C, F)

    z = model.encode(psd_norm).detach().cpu()  # (C, latent)
    z = z.flatten() # (C*latent)
    return z

def train(model, train_loader, val_loader, device, sfreq: float, epochs: int = 100, patience: int = 5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
    wandb.watch(model, log="gradients", log_freq=100)
    best_val_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    min_delta = 0.0
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
        avg_train = total_loss / max(1, n_steps)
        wandb.log({"epoch": epoch, "train/loss": avg_train}, step=epoch)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_steps = 0
            for batch in val_loader:
                psd, freqs = _compute_welch_psd(batch, sfreq)
                B, C, F = psd.shape
                inputs = psd.reshape(B * C, F).to(device)
                mu = inputs.mean(dim=1, keepdim=True)
                std = inputs.std(dim=1, keepdim=True).clamp_min(1e-8)
                inputs = (inputs - mu) / std
                recon = model(inputs)
                if val_steps == 1:  # log once per epoch
                    _wandb_log_recon_example(inputs.detach().cpu(), recon.detach().cpu(), torch.as_tensor(freqs), tag="val/recon_example", step=epoch)
                val_total += float(criterion(recon, inputs).item())
                val_steps += 1
            val_loss = val_total / max(1, val_steps)
            print(f"Epoch {epoch} val loss: {val_loss:.6f}")
            wandb.log({"epoch": epoch, "val/loss": val_loss}, step=epoch)

            # Early stopping logic
            if best_val_loss is None or val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} with best val loss {best_val_loss:.6f}")
                    break
            if best_val_loss is not None:
                wandb.log({"best/val_loss": best_val_loss}, step=epoch)
        model.train()
        # If we broke from early stopping inside validation, exit outer loop as well
        if bad_epochs >= patience:
            break

    # Restore best model parameters if available
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_val_loss is not None:
        wandb.summary["best_val_loss"] = best_val_loss
        

if __name__ == "__main__":
    set_seed()
    device = get_device()
    run = wandb.init(
        project="psd-ae",
        name=f"psd-ae_latent{8}",
        config={
            "seed": SEED,
            "latent_dim": 8,
            "optimizer": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 16,
            "patience": 5,
        },
    )
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

    wandb.config.update({"sfreq": float(dataset.sfreq), "n_train": len(train_ds), "n_val": len(val_ds)})

    train(model, train_loader, val_loader, device=device, sfreq=dataset.sfreq)
    wandb.log({"training/finished": 1})

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    save_path = Path(f"models/psd_ae_{latent_dim}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "freqs": freqs.numpy(),
        "latent_dim": latent_dim,
        "input_dim": input_dim,
    }, str(save_path))

    artifact = wandb.Artifact(f"psd-ae-{latent_dim}", type="model")
    artifact.add_file(str(save_path))
    wandb.log_artifact(artifact)

    wandb.finish()