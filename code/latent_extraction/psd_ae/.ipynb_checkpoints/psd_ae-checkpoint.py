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
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the utils directory to the Python path
utils_path = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from gen_dataset import TUHFIF60sDataset
from util import compute_psd_from_raw, PSD_CALCULATION_PARAMS, compute_psd_from_array, normalize_psd

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


def _plot_recon_example(inputs: torch.Tensor, recon: torch.Tensor, freqs: torch.Tensor, *, path: Path) -> None:
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
    fig.savefig(str(path))
    plt.close(fig)

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
    print(f"Using checkpoint {candidates[0]}")
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

    # Torch 2.6 defaults to weights_only=True which can break older checkpoints
    try:
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch without weights_only kwarg
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
    # Use unified PSD computation (no normalization here; handled below)
    psd_avg_np = compute_psd_from_raw(raw, calculate_average=True, normalize=True)  # (F,)
    psd_avg = torch.from_numpy(psd_avg_np.astype(np.float32)).unsqueeze(0).to(device)  # (1, F)
    

    z = model.encode(psd_avg)
    return z.detach().cpu().numpy().flatten()


@torch.no_grad()
def extract_psd_ae_channel(raw: mne.io.BaseRaw, *, device: Union[str, torch.device] = "cpu",
                           model: Optional[PSDAE] = None, ckpt_path: Optional[str] = None) -> torch.Tensor:
    """Extract a single latent vector by encoding each channel PSD then averaging latents.

    Each channel vector is normalized independently before encoding to match training.
    """
    model = model or get_psd_ae_model(device=device, ckpt_path=ckpt_path)
    # Use unified PSD computation for all channels (C, F)
    psd_np = compute_psd_from_raw(raw, calculate_average=False, normalize=True)  # (C, F)
    psd_t = torch.from_numpy(psd_np.astype(np.float32)).to(device)  # (C, F) on device
    
    z = model.encode(psd_t).detach().cpu().numpy()  # (C, latent)
    z = z.flatten() # (C*latent)
    return z

def train(model, train_loader, val_loader, device, sfreq: float, epochs: int = 100, patience: int = 5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()
    # Precompute Welch parameters and frequency bins once
    n_fft = int(PSD_CALCULATION_PARAMS["n_fft"])
    n_overlap = int(PSD_CALCULATION_PARAMS["n_overlap"])
    n_per_seg = int(PSD_CALCULATION_PARAMS["n_per_seg"])
    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 1.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", 45.0))
    seg_len = int(PSD_CALCULATION_PARAMS["segment_length"] * sfreq)
    dummy = np.zeros(seg_len, dtype=np.float32)
    _, freqs_np = mne.time_frequency.psd_array_welch(
        dummy[None, :],
        sfreq=float(sfreq),
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        average="mean",
        verbose=False,
        fmin=fmin,
        fmax=fmax,
    )
    freqs_t = torch.from_numpy(freqs_np.astype(np.float32))
    best_val_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    min_delta = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        n_steps = 0
        for batch in tqdm(train_loader):
            # batch: (B, C, T) time-domain
            B, C, T = batch.shape
            x_np = batch.detach().cpu().numpy().astype(np.float32)
            x2d = x_np.reshape(B * C, T)
            psd_2d, _ = mne.time_frequency.psd_array_welch(
                x2d,
                sfreq=float(sfreq),
                n_fft=n_fft,
                n_overlap=n_overlap,
                n_per_seg=n_per_seg,
                average="mean",
                verbose=False,
                fmin=fmin,
                fmax=fmax,
            )  # (B*C, F)
            # Normalize per vector (log10 + z-score)
            psd_2d_norm = normalize_psd(psd_2d.astype(np.float32))
            inputs = torch.from_numpy(psd_2d_norm).to(device)

            optimizer.zero_grad()
            recon = model(inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1
        print(f"Epoch {epoch} loss: {total_loss / max(1, n_steps):.6f}")
        avg_train = total_loss / max(1, n_steps)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_steps = 0
            for batch in val_loader:
                B, C, T = batch.shape
                x_np = batch.detach().cpu().numpy().astype(np.float32)
                x2d = x_np.reshape(B * C, T)
                psd_2d, _ = mne.time_frequency.psd_array_welch(
                    x2d,
                    sfreq=float(sfreq),
                    n_fft=n_fft,
                    n_overlap=n_overlap,
                    n_per_seg=n_per_seg,
                    average="mean",
                    verbose=False,
                    fmin=fmin,
                    fmax=fmax,
                )  # (B*C, F)
                psd_2d_norm = normalize_psd(psd_2d.astype(np.float32))
                inputs = torch.from_numpy(psd_2d_norm).to(device)
                recon = model(inputs)
                if val_steps == 1:  # save once per epoch
                    Path("plots").mkdir(exist_ok=True)
                    _plot_recon_example(inputs.detach().cpu(), recon.detach().cpu(), freqs_t.detach().cpu(), path=Path("plots/val_recon_example.png"))
                val_total += float(criterion(recon, inputs).item())
                val_steps += 1
            val_loss = val_total / max(1, val_steps)
            print(f"Epoch {epoch} val loss: {val_loss:.6f}")
            

            # Early stopping logic
            if best_val_loss is None or val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                 # Save model
                Path("models").mkdir(parents=True, exist_ok=True)
                save_path = Path(f"models/psd_ae_{latent_dim}.pth")
                torch.save({
                    "state_dict": model.state_dict(),
                    "freqs": torch.from_numpy(freqs_np.astype(np.float32)),
                    "latent_dim": latent_dim,
                    "input_dim": input_dim,
                }, str(save_path))
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} with best val loss {best_val_loss:.6f}")
                    break
            
        model.train()
        # If we broke from early stopping inside validation, exit outer loop as well
        if bad_epochs >= patience:
            break

    # Restore best model parameters if available
    if best_state is not None:
        model.load_state_dict(best_state)
    
        

if __name__ == "__main__":
    set_seed()
    device = get_device()
    print("[INFO] Starting PSD-AE training run")
    # Load dataset (time-domain segments)
    latent_dim = 8
    batch_size = 512    
    dataset = TUHFIF60sDataset("/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/train_epochs.pkl")
    print(f"Loaded {len(dataset)} files")

    # Determine input_dim from actual PSD frequency bins used (respects fmin/fmax)
    seg_len = int(PSD_CALCULATION_PARAMS["segment_length"] * dataset.sfreq)
    dummy = np.zeros(seg_len, dtype=np.float32)
    _, freqs_np = compute_psd_from_array(dummy, sfreq=dataset.sfreq, return_freqs=True, normalize=False)
    input_dim = int(freqs_np.shape[0])
   
    # Create model for per-channel PSD vectors
    model = PSDAE(input_dim=input_dim, latent_dim=latent_dim).to(device)

    from torch.utils.data import random_split
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    train_ds, val_ds = random_split(
        dataset, [n - n_val, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))

    print(f"num train samples={len(train_ds)} num val samples={len(val_ds)}")

    train(model, train_loader, val_loader, device=device, sfreq=dataset.sfreq)
    print("[INFO] Training finished")

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    save_path = Path(f"models/psd_ae_{latent_dim}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "freqs": torch.from_numpy(freqs_np.astype(np.float32)),
        "latent_dim": latent_dim,
        "input_dim": input_dim,
    }, str(save_path))

    print(f"[INFO] Saved model to {save_path}")
    