
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import random
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
import numpy as np
import mne
from torch.utils.data import random_split

import torch.nn.functional as F
from math import ceil



# === EEG Dataset Class ===
class EEGSegmentDataset(Dataset):
    def __init__(self, bids_root, segment_len_samples=7680):
        self.segment_len = segment_len_samples
        self.raw_segments = []

        subject_dirs = [p for p in bids_root.glob("sub-*") if p.is_dir()]

        for subj_dir in subject_dirs:
            vhdr_files = list(subj_dir.rglob("*_eeg.vhdr"))
            for vhdr_file in vhdr_files:
                try:
                    subject = vhdr_file.parts[-4].replace("sub-", "")
                    session = vhdr_file.parts[-3].replace("ses-", "")
                    task = vhdr_file.name.split("_task-")[1].split("_")[0]

                    bids_path = BIDSPath(
                        subject=subject,
                        session=session,
                        task=task,
                        datatype="eeg",
                        root=bids_root
                    )

                    raw = read_raw_bids(bids_path, verbose=False)
                    raw.load_data()
                    sfreq = raw.info['sfreq']
                    raw.crop(0, 60.0 - 1/sfreq)  # Crop to segment length
                    
                    # ✅ Expect exactly 7680 samples after preprocessing
                    if raw.n_times != self.segment_len:
                        print(f"⚠️ Skipping {vhdr_file.name}: got {raw.n_times} samples, expected {self.segment_len}")
                        continue

                    segment = raw.get_data()  # shape: (channels, 7680)
                    self.raw_segments.append(segment.astype(np.float32))

                except Exception as e:
                    print(f"⚠️ Skipping file {vhdr_file.name}: {e}")

    def __len__(self):
        return len(self.raw_segments)

    def __getitem__(self, idx):
        segment = self.raw_segments[idx]
        segment = (segment - np.mean(segment)) / np.std(segment)  # z-score normalization
        return torch.tensor(segment, dtype=torch.float32)
    
    
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


def train_autoencoder(model, train_loader, val_loader, optimizer, criterion, n_epochs=20, device='cpu'):
    model.to(device)

    # Add ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch = batch.to(device).float()
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device).float()
                output = model(batch)
                loss = criterion(output, batch)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

def autoencoder_main(name_dataset="harvard"):
    """ Main function to train the Conv1D autoencoder on EEG data. """

    if name_dataset == "harvard":        
        bids_root_small = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/bids_root_small_clean")
        segment_len_sec = 60
        sample_rate = 128
        segment_len_samples = segment_len_sec * sample_rate 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = EEGAutoEncoder(chans=19, latent_dim=64, fixed_len=segment_len_samples)
        model = model.to(device)
 
        criterion = nn.MSELoss()
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
        train_autoencoder(model, train_loader, val_loader, optimizer, criterion, n_epochs=20, device=device)
        torch.save(model.state_dict(), "conv1d_autoencoder.pth")
        
        
        
    
    else:
        raise ValueError(f"Unknown dataset: {name_dataset}. Choose 'harvard' or 'tuh'.")
    
    
if __name__ == "__main__":
    autoencoder_main(name_dataset="harvard")  # Change to "tuh" if needed
    # autoencoder_main(name_dataset="tuh")  # Uncomment to run on TUH
    