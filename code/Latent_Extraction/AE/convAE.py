
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
            nn.LeakyReLU(0.1)
        )
        self.e2 = nn.Sequential(          # L/2 → L/4
            nn.Conv1d(chs[0], chs[1], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(chs[1]),
            nn.LeakyReLU(0.1)
        )
        self.e3 = nn.Sequential(          # L/4 → L/8
            nn.Conv1d(chs[1], chs[2], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(chs[2]),
            nn.LeakyReLU(0.1)
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
            nn.LeakyReLU(0.1)
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose1d(chs[1]*2, chs[0], 5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(chs[0]),
            nn.LeakyReLU(0.1)
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

def train_autoencoder(model, train_loader, val_loader, optimizer, criterion, n_epochs=20, device='cpu'):
    model.to(device)

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
        
        

def autoencoder_main(name_dataset="harvard"):
    """ Main function to train the Conv1D autoencoder on EEG data. """

    if name_dataset == "harvard":        
        bids_root_small = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/bids_root_small_clean")
        segment_len_sec = 60
        sample_rate = 128
        segment_len_samples = segment_len_sec * sample_rate 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Conv1DAutoencoder(n_channels=19, fixed_len=segment_len_samples, latent_dim=16)
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
        
    else if name_dataset == "tuh":
        #tuh dataset is set of fif files, so we need to load them differently
        fif_folder = Path("/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean")
        
        
        
        
    
    else:
        raise ValueError(f"Unknown dataset: {name_dataset}. Choose 'harvard' or 'tuh'.")
    
    
if __name__ == "__main__":
    # autoencoder_main(name_dataset="harvard-eeg")  # Change to "tuh" if needed
    autoencoder_main(name_dataset="tuh")  # Uncomment to run on TUH