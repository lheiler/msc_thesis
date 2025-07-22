
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
import numpy as np
import mne
from mne.io import read_raw_brainvision
from torch.utils.data import random_split

import torch.nn.functional as F

from math import ceil
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts




# === EEG Dataset Class ===
class EEGSegmentDataset(Dataset):
    """
    Lazily loads 60‑s BrainVision recordings.  Each item reads exactly one
    .vhdr header, crops the Raw object BEFORE streaming samples, and returns a
    z‑scored (C, T) float32 tensor.  Faster and lower‑RAM than eager loading.
    """
    def __init__(self, bids_root: Path, segment_len_samples: int = 7680):
        super().__init__()
        self.segment_len = segment_len_samples

        # Collect all *_eeg.vhdr files once; no I/O on the data here
        self.vhdr_files = [
            p for p in bids_root.rglob("*_eeg.vhdr") if p.is_file()
        ]
        if not self.vhdr_files:
            raise RuntimeError(f"No BrainVision .vhdr files found in {bids_root}")

    def __len__(self):
        return len(self.vhdr_files)

    def __getitem__(self, idx: int):
        vhdr = self.vhdr_files[idx]

        # --- Load header only (preload=False) ---
        raw = read_raw_brainvision(str(vhdr), preload=False, verbose=False)

        # Crop BEFORE streaming samples to RAM
        sfreq = raw.info["sfreq"]
        t_max = (self.segment_len - 1) / sfreq   # inclusive end
        raw.crop(tmin=0.0, tmax=t_max)
        raw.load_data(verbose=False)             # now streams only 60 s

        data = raw.get_data().astype(np.float32)  # (channels, T)

        # Sanity‑check length; sometimes header rounding gives ±1 sample
        if data.shape[1] != self.segment_len:
            if data.shape[1] > self.segment_len:
                data = data[:, : self.segment_len]
            else:  # pad with zeros
                pad = self.segment_len - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad)), mode="constant")

        # Z‑score normalisation across the entire segment
        data = (data - data.mean()) / (data.std() + 1e-8)

        return torch.from_numpy(data)
    
# === Conv1D Autoencoder Class ===    
class Conv1DAutoencoder(nn.Module):
    """
    U-Net-style 1-D convolutional auto-encoder for raw EEG.

    Args
    ----
    n_channels : int
        Number of EEG channels (default 19).
    latent_dim : int
        Dimensionality of bottleneck.
    fixed_len : int or None
        • int  → expect exactly this many samples (fastest).  
        • None → accept arbitrary length; an adaptive pool is inserted.
    """
    def __init__(self, n_channels=19, latent_dim=32, fixed_len=7680):
        super().__init__()
        chs = [32, 64, 128]       # channels at each down-sampling step

        # ---------- Encoder ----------
        self.e1 = nn.Sequential(          # L → L/2
            nn.Conv1d(n_channels, chs[0], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(chs[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.e2 = nn.Sequential(          # L/2 → L/4
            nn.Conv1d(chs[0], chs[1], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(chs[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.e3 = nn.Sequential(          # L/4 → L/8
            nn.Conv1d(chs[1], chs[2], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(chs[2]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        # If the input length is fixed we can pre-compute the flattened size
        if fixed_len is not None:
            dummy = torch.zeros(1, n_channels, fixed_len)
            with torch.no_grad():
                enc_out = self._encode(dummy)[0]       # (B, C, T)
            self.flat_dim = enc_out.numel()
            self.adapt_pool = None
        else:
            # Variable length → use AdaptiveAvgPool1d to enforce a fixed T=32
            self.adapt_pool = nn.AdaptiveAvgPool1d(32)
            self.flat_dim = chs[-1] * 32

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # ---------- Decoder ----------
        # Mirrors encoder, but note the order: transposed conv → BN → LeakyReLU
        self.d3 = nn.Sequential(
            nn.ConvTranspose1d(chs[2], chs[1], 5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(chs[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose1d(chs[1]*2, chs[0], 5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(chs[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.d1 = nn.Sequential(
            nn.ConvTranspose1d(chs[0]*2, n_channels, 5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()                           # keep raw EEG in (-1, 1)
        )

    # ---- Helper to run the three enc convs and return their outputs ----
    def _encode(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        return s1, s2, s3

    def forward(self, x):
        # ---------- Encoder ----------
        skips = self._encode(x)          # tuple of 3 tensors

        z = skips[-1]
        if self.adapt_pool is not None:
            z = self.adapt_pool(z)       # (B, 128, 32)

        z = z.flatten(1)                 # → (B, flat_dim)
        z = self.fc_enc(z)               # bottleneck

        # ---------- Decoder ----------
        y = self.fc_dec(z).view_as(skips[-1])   # reshape to (B, 128, T/8)
        y = self.d3(y)

        # ----- Skip connections: concat along channel axis -----
        y = self.d2(torch.cat([y, skips[1]], dim=1))
        y = self.d1(torch.cat([y, skips[0]], dim=1))
        return y

 # === Advanced EEG Auto‑Encoder Components ==================================
class DepthwiseSpatialStem(nn.Module):
    """
    Depthwise‑separable 1‑D convolution that first learns per‑channel filters
    and then mixes them across channels. Operates on tensors shaped
    (batch, channels, time).
    """
    def __init__(self, in_chans: int, out_chans: int = 32):
        super().__init__()
        self.depthwise = nn.Conv1d(in_chans, in_chans, kernel_size=1,
                                   groups=in_chans, bias=False)
        self.pointwise = nn.Conv1d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_chans)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class ResidualDilatedBlock(nn.Module):
    """
    Multi‑branch residual block with dilated convolutions for multi‑scale
    temporal context.  Each branch uses a different (kernel size, dilation)
    pair, the outputs are concatenated, reduced, and added to a shortcut path.
    """
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 kernel_sizes=(3, 5, 9),
                 dilations=(1, 2, 4),
                 stride: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), \
            "kernel_sizes and dilations must be the same length"
        branches = []
        for k, d in zip(kernel_sizes, dilations):
            pad = (k // 2) * d
            branches.append(
                nn.Conv1d(
                    in_chans,
                    out_chans,
                    kernel_size=k,
                    stride=stride,
                    padding=pad,
                    dilation=d,
                    bias=False,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.bn = nn.BatchNorm1d(out_chans * len(branches))
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)
        self.reduce = nn.Conv1d(out_chans * len(branches), out_chans, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_chans)

        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_chans, out_chans, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_chans),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.act(self.bn(out))
        out = self.drop(out)
        out = self.bn2(self.reduce(out))
        return self.act(out + self.shortcut(x))


# === Channel-wise Squeeze-and-Excitation ==================================
class ChannelSE(nn.Module):
    """Squeeze‑and‑Excitation for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.avg(x))
        return x * scale


class EEGAutoEncoder(nn.Module):
    """
    Advanced multi‑scale auto‑encoder for 19‑channel EEG.
    Encoder: spatial stem → three residual‑dilated blocks (stride‑2 each)
    Bottleneck: 2‑layer Transformer encoder + linear projection to latent dim
    Decoder: symmetric up‑sampling with skip connections.
    Assumes fixed_len samples so shapes can be inferred.
    """
    def __init__(self, chans: int = 19, latent_dim: int = 64, fixed_len: int = 7680):
        super().__init__()
        self.fixed_len = fixed_len

        # ----- Encoder -----------------------------------------------------
        self.stem = DepthwiseSpatialStem(chans, 32)            # (B, 32, L)
        self.enc1 = ResidualDilatedBlock(32,  64, stride=2)    # L/2
        self.enc2 = ResidualDilatedBlock(64, 128, stride=2)    # L/4
        self.enc3 = ResidualDilatedBlock(128, 256, stride=2)   # L/8
        # Channel attention
        self.se = ChannelSE(256)

        # ----- Transformer Bottleneck -------------------------------------
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                batch_first=True,
            ),
            num_layers=2,
        )

        self.T_reduced = ceil(fixed_len / 8)      # after three stride‑2 downsamples
        self.flat_dim = 256 * self.T_reduced

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # ----- Decoder -----------------------------------------------------
        self.up3  = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResidualDilatedBlock(256, 128, kernel_sizes=(3,), dilations=(1,), dropout=0.1)

        self.up2  = nn.ConvTranspose1d(128,  64, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResidualDilatedBlock(128,  64, kernel_sizes=(3,), dilations=(1,), dropout=0.1)

        self.up1  = nn.ConvTranspose1d( 64,  32, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResidualDilatedBlock( 64,  32, kernel_sizes=(3,), dilations=(1,), dropout=0.1)

        self.final_conv = nn.Conv1d(32, chans, kernel_size=1)
        self.out_act    = nn.Tanh()

    def forward(self, x):
        # ---------------- Encoder ----------------
        s1 = self.stem(x)      # (B, 32, L)
        s2 = self.enc1(s1)     # (B, 64, L/2)
        s3 = self.enc2(s2)     # (B,128, L/4)
        s4 = self.enc3(s3)     # (B,256, L/8)
        s4 = self.se(s4)          # channel‑wise re‑weighting

        # Transformer expects (B, T, C)
        t = self.transformer(s4.permute(0, 2, 1)).permute(0, 2, 1)  # (B,256,T/8)

        # Latent projection
        z = self.fc_enc(t.flatten(1))                               # (B, latent_dim)
        y = self.fc_dec(z).view_as(t)                               # (B,256,T/8)

        # ---------------- Decoder ----------------
        y = self.up3(y)
        y = self.dec3(torch.cat([y, s3], dim=1))

        y = self.up2(y)
        y = self.dec2(torch.cat([y, s2], dim=1))

        y = self.up1(y)
        y = self.dec1(torch.cat([y, s1], dim=1))

        return self.out_act(self.final_conv(y))


# === Loss helpers =========================================================
def spectral_loss(x_hat: torch.Tensor, x: torch.Tensor,
                  n_fft: int = 256, hop: int = 128) -> torch.Tensor:
    """
    Compute log‑spectral mean‑squared error between two EEG segments.

    Both inputs are expected to be shaped (B, C, T). Since
    ``torch.stft`` only accepts 1‑D (T) or 2‑D (B, T) tensors, the
    batch and channel dimensions are flattened to (B*C, T) before
    calling ``stft``.  A Hann window is used to minimise spectral
    leakage.

    Parameters
    ----------
    x_hat : torch.Tensor
        Reconstructed signal of shape (B, C, T).
    x : torch.Tensor
        Target signal of shape (B, C, T).
    n_fft : int
        FFT size.
    hop : int
        Hop length between STFT frames.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # ----- Flatten to 2‑D: (B*C, T) -----
    B, C, T = x.shape
    x_hat_flat = x_hat.reshape(B * C, T)
    x_flat     = x.reshape(B * C, T)

    # ----- Hann window to reduce leakage -----
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)

    # ----- STFT -----
    Xh = torch.stft(x_hat_flat, n_fft=n_fft, hop_length=hop,
                    window=window, return_complex=True)
    X  = torch.stft(x_flat,     n_fft=n_fft, hop_length=hop,
                    window=window, return_complex=True)

    # ----- Log‑spectral MSE -----
    return F.mse_loss(torch.log1p(torch.abs(Xh)),
                      torch.log1p(torch.abs(X)))

def mixed_loss(x_hat: torch.Tensor,
               x: torch.Tensor,
               mask: torch.Tensor | None = None,
               alpha: float = 0.7) -> torch.Tensor:
    """
    Combined reconstruction loss.

    * **Time‑domain MSE** is evaluated **only on the masked positions**
      (if a boolean `mask` tensor is supplied). This encourages the model
      to in‑fill the occluded parts without being penalised for simply
      copying the visible input.
    * **Log‑spectral MSE** is always evaluated on the full sequences to
      preserve frequency‑domain consistency across the entire segment.

    Parameters
    ----------
    x_hat : torch.Tensor
        Model output of shape ``(B, C, T)``.
    x : torch.Tensor
        Ground‑truth target of shape ``(B, C, T)``.
    mask : torch.Tensor | None
        Boolean tensor (same shape as ``x``) indicating the time samples
        that were masked out during input corruption.  If ``None``, the
        MSE term is computed on all samples.
    alpha : float
        Weighting factor for the time‑domain MSE term.  The spectral term
        is weighted by ``(1 - alpha)``.

    Returns
    -------
    torch.Tensor
        Scalar loss value suitable for ``loss.backward()``.
    """
    if mask is not None:
        mse = F.mse_loss(x_hat[mask], x[mask])
    else:
        mse = F.mse_loss(x_hat, x)

    spec = spectral_loss(x_hat, x)
    return alpha * mse + (1 - alpha) * spec


# === Masked reconstruction helper ========================================
def apply_time_mask(x: torch.Tensor, mask_ratio: float = 0.3,
                    block_size: int = 256):
    """
    Zero‑out random time blocks (size = block_size samples) of each segment.
    Returns masked input and boolean mask of *masked* positions.
    """
    B, C, T = x.shape
    num_blocks = int(T * mask_ratio // block_size)
    mask = torch.zeros_like(x, dtype=torch.bool)

    for b in range(B):
        idx = torch.randperm(T // block_size)[:num_blocks]
        for i in idx:
            s, e = i*block_size, (i+1)*block_size
            x[b, :, s:e] = 0.0
            mask[b, :, s:e] = True
    return x, mask


def train_autoencoder(model, train_loader, val_loader, optimizer,
                      n_epochs: int = 40, device: str = 'cpu',
                      early_stop_patience: int = 8,
                      mask_ratio: float = 0.3):
    """
    Joint masked‑reconstruction training with mixed MSE + spectral loss,
    cosine warm restarts, aggressive ReduceLROnPlateau and early stopping.
    """
    model.to(device)

    # LR schedulers
    plateau = ReduceLROnPlateau(optimizer, mode='min',
                                factor=0.5, patience=1,
                                min_lr=1e-5)
    cosine  = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(n_epochs):
        # ----------- Training -----------
        model.train()
        total_train = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch = batch.to(device).float()
            masked, mask_bool = apply_time_mask(batch.clone(),
                                                mask_ratio=mask_ratio)

            optimizer.zero_grad()
            out = model(masked)
            loss = mixed_loss(out, batch, mask_bool)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)

        # ----------- Validation -----------
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device).float()
                out   = model(batch)
                loss  = mixed_loss(out, batch)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)

        # Scheduler updates
        plateau.step(avg_val)
        cosine.step(epoch + 1)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train:.4f} "
              f"| Val Loss: {avg_val:.4f}")

        # Early stopping
        if avg_val + 1e-4 < best_val:
            best_val = avg_val
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop_patience:
                print("Early stopping triggered.")
                break

def autoencoder_main(name_dataset="harvard"):
    """ Main function to train the Conv1D autoencoder on EEG data. """

    if name_dataset == "harvard":        
        bids_root_small = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/bids_root_small_clean")
        segment_len_sec = 60
        sample_rate = 128
        segment_len_samples = segment_len_sec * sample_rate 

        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
        print("we quick out here")
        model = EEGAutoEncoder(chans=19, latent_dim=64, fixed_len=segment_len_samples)
        print("not anymore")
        model = model.to(device)
 
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # ✅ Use the small subset path here!
        dataset = EEGSegmentDataset(bids_root=bids_root_small, segment_len_samples=segment_len_samples)
        val_ratio = 0.2
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # ✅ Use correct DataLoader variable name
        train_autoencoder(model, train_loader, val_loader, optimizer,
                          n_epochs=40, device=device,
                          early_stop_patience=8, mask_ratio=0.3)
        torch.save(model.state_dict(), "conv1d_autoencoder.pth")
        
        
        
    
    else:
        raise ValueError(f"Unknown dataset: {name_dataset}. Choose 'harvard' or 'tuh'.")
    
    
if __name__ == "__main__":
    autoencoder_main(name_dataset="harvard")  # Change to "tuh" if needed
    # autoencoder_main(name_dataset="tuh")  # Uncomment to run on TUH
    
