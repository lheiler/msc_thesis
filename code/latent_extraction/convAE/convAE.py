#convAE train.py
# Plain 1D convolutional autoencoder for raw EEG windows
# Inputs: x shaped (B, C, T)
# Output: reconstructed x̂ shaped (B, C, T)

import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional

import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm

class ConvAE1D(nn.Module):
    """
    Simple 1D Conv Autoencoder for multi-channel EEG.

    Encoder: Conv1d blocks with stride-2 downsampling along time.
    Decoder: mirrored ConvTranspose1d blocks with stride-2 upsampling.

    This mixes information across channels (Conv1d in_channels=C), which is a
    common baseline for denoising/compression on raw EEG windows.
    """

    def __init__(
        self,
        n_channels: int,
        input_window_samples: int,
        latent_dim: int = 128,
        enc_channels: Tuple[int, int, int] = (64, 128, 256),
        kernel_size: int = 8,
        stride: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 0, "kernel_size should be even to keep padding clean"
        self.n_channels = n_channels
        self.T = input_window_samples

        # ---------------- Encoder ----------------
        layers: List[nn.Module] = []
        in_ch = n_channels
        L = input_window_samples
        for out_ch in enc_channels:
            pad = kernel_size // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
            L = (L + 2 * pad - kernel_size) // stride + 1  # conv1d output length
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, input_window_samples)
            h = self.encoder(dummy)
            self.enc_feat_len = h.shape[-1]
            self.enc_feat_ch = h.shape[1]
            self.flat_dim = self.enc_feat_ch * self.enc_feat_len

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # ---------------- Decoder ----------------
        dec_layers: List[nn.Module] = []
        dec_channels = list(enc_channels[::-1])
        in_ch = dec_channels[0]
        for i, out_ch in enumerate(dec_channels[1:] + [n_channels]):
            dec_layers += [
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=(self._output_padding(self.T, i + 1, kernel_size, stride)),
                    bias=False,
                ),
            ]
            # Add BN+ReLU on all but the final layer
            if i < len(dec_channels) - 1:
                dec_layers += [nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)
        self.out_act = nn.Tanh()

    # ---- helpers ----
    def _output_padding(self, target_len: int, num_ups: int, k: int, s: int) -> int:
        """
        Compute output_padding to exactly recover target_len after a stack of
        num_ups transposed-conv layers with kernel k and stride s.
        Assumes same padding as in the encoder.
        """
        # Simulate downsampling num_ups times
        L = target_len
        pad = k // 2
        for _ in range(num_ups):
            L = (L + 2 * pad - k) // s + 1
        # Now invert once to decide output_padding for this stage
        # We choose 0 or 1 so that repeated upsampling lands exactly on target_len
        # This function is called per stage to stabilize lengths.
        # For simplicity return 0; lengths are corrected cumulatively in decode.
        return 0

    # ---- API ----
    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z = self.fc_mu(h.flatten(start_dim=1))
        return z

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_dec(z).view(z.size(0), self.enc_feat_ch, self.enc_feat_len)
        y = self.decoder(h)
        # Correct any 1-sample mismatch due to padding/stride rounding
        if y.shape[-1] != self.T:
            y = F.interpolate(y, size=self.T, mode="nearest")
        return self.out_act(y)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        return self.decode(z)


# ---- minimal training loop ----

def recon_loss(y: Tensor, x: Tensor) -> Tensor:
    return F.mse_loss(y, x)


def train(
    model: ConvAE1D,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_interval: int = 50,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in tqdm(range(1, epochs + 1)):
        model.train()
        total = 0.0
        n = 0
        for i, batch in enumerate(train_loader):
            x = batch.to(device).float()  # (B, C, T)
            y = model(x)
            loss = recon_loss(y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            if log_interval and (i + 1) % log_interval == 0:
                print(f"epoch {ep} step {i+1}: train_mse={total / n:.6f}")

        if val_loader is not None:
            model.eval()
            v_total = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device).float()  # (B, C, T)
                    y = model(x)
                    v_total += recon_loss(y, x).item() * x.size(0)
                    v_n += x.size(0)
            print(f"epoch {ep}: val_mse={v_total / max(1, v_n):.6f}")

    return model


# Optional builder for convenience

def build_model(n_channels: int, input_window_samples: int, latent_dim: int = 128) -> ConvAE1D:
    return ConvAE1D(
        n_channels=n_channels,
        input_window_samples=input_window_samples,
        latent_dim=latent_dim,
    )


# ---------------- Inference helpers: load + extract ----------------

def get_conv_ae_model(
    device: torch.device,
    model_path: str,
    *,
    n_channels: int = 19,
    input_window_samples: int = 60 * 128,
    latent_dim: int = 128,
) -> ConvAE1D:
    """
    Load a trained ConvAE1D model from a state_dict file.

    Parameters
    ----------
    device : torch.device
        Target device.
    model_path : str
        Path to a torch state_dict saved via model.state_dict().
    n_channels : int
        Number of EEG channels expected by the model.
    input_window_samples : int
        Fixed window length in samples (default 60 s at 128 Hz).
    latent_dim : int
        Size of latent vector.
    """
    model = ConvAE1D(
        n_channels=n_channels,
        input_window_samples=input_window_samples,
        latent_dim=latent_dim,
    )
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_conv_ae(x_raw, *, device: torch.device, model: ConvAE1D) -> torch.Tensor:
    """
    Extract a single latent vector from an MNE Raw by windowing
    into non-overlapping segments of length model.T and averaging
    the latent codes.

    Assumptions
    -----------
    - `x_raw` already contains exactly the 19 EEG channels in the
      order enforced upstream (Fp1..Fz). Channel selection happens
      before calling this function.
    - Resamples to 128 Hz to match the default model configuration.

    Returns
    -------
    torch.Tensor of shape (latent_dim,)
    """
    assert isinstance(x_raw, mne.io.BaseRaw), "expected MNE Raw"

    # Resample to 128 Hz to match `input_window_samples = 60*128` default
    target_sfreq = 128.0
    if not np.isclose(x_raw.info["sfreq"], target_sfreq):
        x = x_raw.copy().resample(target_sfreq, npad="auto")
    else:
        x = x_raw

    data = x.get_data(picks="eeg", reject_by_annotation="omit")  # (C, T_total)
    C, T_total = data.shape
    T = model.T

    if T_total < T:
        raise RuntimeError(f"recording shorter than model window: {T_total} < {T}")

    # Build non-overlapping windows
    n_win = T_total // T
    if n_win == 0:
        raise RuntimeError("no full windows available")

    z_list = []
    for i in range(n_win):
        seg = data[:, i * T:(i + 1) * T]
        xt = torch.from_numpy(seg).to(device=device, dtype=torch.float32).unsqueeze(0)  # (1,C,T)
        z = model.encode(xt).squeeze(0)  # (latent_dim,)
        z_list.append(z)

    z_mean = torch.stack(z_list, dim=0).mean(dim=0)
    return z_mean.cpu()


# ---------------- Dataset for streaming Raw → (C,T) tensors ----------------

EEG_19 = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Cz", "Pz", "Fz",
]


class EegFixedLengthDataset(Dataset):
    """
    Loads one fixed-length (C, T) tensor per file, assuming each recording is 60s.
    No windowing: each file → one item.
    """

    def __init__(
        self,
        file_paths: List[str],
        *,
        target_sfreq: float = 128.0,
        duration_s: float = 60.0,
        channels: Optional[List[str]] = None,
        fmin: float = 1.0,
        fmax: float = 45.0,
    ) -> None:
        self.file_paths = list(file_paths)
        self.target_sfreq = float(target_sfreq)
        self.duration_s = float(duration_s)
        self.channels = list(channels or EEG_19)
        self.fmin = float(fmin)
        self.fmax = float(fmax)

        self.window_len = int(round(self.target_sfreq * self.duration_s))

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tensor:
        fp = self.file_paths[idx]
        raw = mne.io.read_raw_fif(fp, preload=True, verbose=False)

        # Drop A1/A2 if present; pick 19 EEG channels
        if "A1" in raw.ch_names:
            raw.drop_channels(["A1"]) 
        if "A2" in raw.ch_names:
            raw.drop_channels(["A2"]) 
        raw.pick_channels(self.channels)

        # Resample then band-pass filter
        if not np.isclose(raw.info["sfreq"], self.target_sfreq):
            raw.resample(self.target_sfreq, npad="auto")
        raw.filter(self.fmin, self.fmax, verbose=False)

        data = raw.get_data(picks="eeg", reject_by_annotation="omit")  # (C, T_total)
        C, T_total = data.shape

        # Strict assumption: at least 60s worth of samples
        if T_total < self.window_len:
            raise RuntimeError(f"Recording shorter than expected 60s: {T_total} < {self.window_len} samples after processing for {fp}")

        seg = data[:, : self.window_len]
        x = seg.astype(np.float32)
        return torch.from_numpy(x)


if __name__ == "__main__":
    # Find all FIF files
    all_files = glob.glob("/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/train/**/*.fif", recursive=True)
    # Split files (simple split)
    split = int(0.9 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]

    # Streaming datasets → tensors (C, T)
    target_sfreq = 128.0
    duration_s = 60.0
    train_ds = EegFixedLengthDataset(train_files, target_sfreq=target_sfreq, duration_s=duration_s, channels=EEG_19, fmin=1.0, fmax=45.0)
    val_ds = EegFixedLengthDataset(val_files, target_sfreq=target_sfreq, duration_s=duration_s, channels=EEG_19, fmin=1.0, fmax=45.0)

    # DataLoaders
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_model(n_channels=19, input_window_samples=int(target_sfreq * duration_s), latent_dim=128)
    model = train(model, train_loader, val_loader, device=device)
    torch.save(model.state_dict(), "convAE_model.pt")
    