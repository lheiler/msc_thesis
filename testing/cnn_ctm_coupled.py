# Prototype coupled cortico-thalamic model (CTM) + classifier training
# Author: AI assistant
#
# This script implements an end-to-end differentiable pipeline that fits a per-channel
# computational brain model (here: a lightweight AR(3) approximation of the CTM) to
# cleaned TUH EEG segments while *simultaneously* training a downstream classifier.
# The fitting loss (PSD match) and classification loss are jointly optimised via
# back-propagation.
#
# Assumptions
# -----------
# 1. The cleaned data are provided as two NumPy files inside a directory (see README):
#       segments.npy   – shape (N, 19, 7680)  [60 s @ 128 Hz]
#       labels.npy     – shape (N,)            [int class labels]
# 2. You have PyTorch ≥2.0 installed (see requirements.txt).
#
# Usage
# -----
#     python prototype_ctm_coupled.py --data_root /path/to/data \
#            --epochs 20 --batch_size 8 --alpha 1.0 --beta 1.0
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import argparse
import time
from typing import Tuple, Optional
from math import pi
import sys
# Data science stack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt  # For diagnostic plots

# EEG I/O
import mne  # Added for reading .fif files


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------


class EEGDataset(Dataset):
    """Dataset for cleaned TUH EEG recordings.

    The class supports two data layouts:

    1. Legacy NumPy blobs::
           data_root/
             ├─ segments.npy   (N, 19, 7680)
             └─ labels.npy     (N,)

    2. Hierarchical folder structure with individual *.fif* files::
           data_root/
             ├─ normal/   *.fif
             └─ abnormal/ *.fif

       The label is inferred from the immediate parent folder name. The first 60 s
       (or a random 60-s chunk) of each recording are extracted, resampled to
       128 Hz, and returned as a 19 × 7680 float-tensor.
    """

    SEG_LEN = 60 * 128  # 60 s @ 128 Hz → 

    def __init__(self, data_root: str, random_segment: bool = True):
        self.random_segment = random_segment

        seg_path = os.path.join(data_root, "segments.npy")
        lab_path = os.path.join(data_root, "labels.npy")

        # ------------------------------------------------------------------
        # Case 1 – pre-generated NumPy arrays
        # ------------------------------------------------------------------
        if os.path.exists(seg_path) and os.path.exists(lab_path):
            # Load arrays fully into memory.
            self.segments = np.load(seg_path)  # (N, 19, SEG_LEN)
            self.labels = np.load(lab_path)    # (N,)

            if self.segments.shape[0] != self.labels.shape[0]:
                raise ValueError("segments and labels count mismatch")

            if self.segments.shape[1] != 19 or self.segments.shape[2] != self.SEG_LEN:
                raise ValueError("Expected shape (N, 19, 7680) for segments.npy")

            self._use_numpy_blobs = True
            return

        # ------------------------------------------------------------------
        # Case 2 – hierarchical folder with .fif files
        # ------------------------------------------------------------------
        self._use_numpy_blobs = False
        self.file_paths: list[str] = []
        self.labels = []

        label_map = {"normal": 0, "abnormal": 1}

        for cls_name, cls_label in label_map.items():
            cls_dir = os.path.join(data_root, cls_name)
            if not os.path.isdir(cls_dir):
                # Skip if split (e.g. eval) does not contain a class
                continue
            for root, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith(".fif"):
                        self.file_paths.append(os.path.join(root, fname))
                        self.labels.append(cls_label)

        if not self.file_paths:
            raise FileNotFoundError(
                "No data found. Provide either segments.npy / labels.npy or a folder with *.fif files under 'normal' / 'abnormal'."
            )

        # Convert to numpy array for easy indexing & compatibility with original code
        self.labels = np.asarray(self.labels, dtype=np.int64)

        # Sort for deterministic order
        self.file_paths, self.labels = zip(*sorted(zip(self.file_paths, self.labels)))
        self.file_paths = list(self.file_paths)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.labels)

    # ------------------------------------------------------------------
    def _load_fif_segment(self, path: str) -> np.ndarray:
        """Load a 19×7680 segment from an EEG recording (.fif)."""
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")

        # Keep EEG channels only and ensure 128 Hz sampling
        # pick exactly the 19 channels that are relevant for the CTM
        ch_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"] 
        raw.pick(ch_names)
        
        if raw.info["sfreq"] != 128.0:
            raw.resample(128.0)
        raw.crop(tmin=0, tmax=60-1/128)

        data = raw.get_data()  # (n_channels, n_samples)

        if data.shape[0] != 19:
            raise ValueError(f"Expected 19 channels, got {data.shape[0]} for {path}")

        n_samples = data.shape[1]
        if n_samples < self.SEG_LEN:
            # Zero-pad short recordings
            out = np.zeros((19, self.SEG_LEN), dtype=np.float32)
            out[:, :n_samples] = data.astype(np.float32)
            return out

        # Choose segment (first or random 60 s)
        if self.random_segment and n_samples > self.SEG_LEN:
            start = np.random.randint(0, n_samples - self.SEG_LEN + 1)
        else:
            start = 0
        end = start + self.SEG_LEN
        return data[:, start:end].astype(np.float32)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self._use_numpy_blobs:
            x = torch.from_numpy(self.segments[idx]).float()
            # Per-segment z-score normalisation (across all values)
            x = (x - x.mean()) / (x.std() + 1e-6)
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y, idx

        # Folder-based dataset
        path = self.file_paths[idx]
        x_np = self._load_fif_segment(path)  # (19, 7680)
        x = torch.from_numpy(x_np)
        # Per-segment z-score normalisation
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, idx


# -----------------------------------------------------------------------------
# Helper to auto-detect train/eval splits and build DataLoaders
# -----------------------------------------------------------------------------

def build_split_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    random_segment_train: bool = False,
    random_segment_eval: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], EEGDataset, Optional[EEGDataset]]:
    """Create train and (optionally) eval DataLoaders from a root folder.

    If ``data_root/train`` exists, it is used for training; otherwise ``data_root``
    itself is used. If ``data_root/eval`` exists, it is used as eval split.

    Returns
    -------
    train_dl, eval_dl, train_ds, eval_ds
    """
    train_dir = os.path.join(data_root, "train")
    eval_dir = os.path.join(data_root, "eval")

    if os.path.isdir(train_dir):
        train_ds = EEGDataset(train_dir, random_segment=random_segment_train)
    else:
        train_ds = EEGDataset(data_root, random_segment=random_segment_train)

    eval_ds = None
    if os.path.isdir(eval_dir):
        eval_ds = EEGDataset(eval_dir, random_segment=random_segment_eval)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    eval_dl = None
    if eval_ds is not None:
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_dl, eval_dl, train_ds, eval_ds


# -----------------------------------------------------------------------------
# CTM analytic constants (from old/cbm_implemenation.py)
# -----------------------------------------------------------------------------
Lx = Ly = 0.5       # metres
k0 = 10.0           # m^-1
# fixed physiological constants
gamma_e = 116.0     # s^-1
r_e     = 0.086     # metres (86 mm)
M = 10              # spatial truncation

# Pre-compute the spatial grid (constant, CPU). We move it to device on demand.
_m = torch.arange(-M, M + 1, dtype=torch.float32)
_kx = 2 * pi * _m[:, None] / Lx  # (21,1)
_ky = 2 * pi * _m[None, :] / Ly  # (1,21)
_k2 = _kx ** 2 + _ky ** 2        # (21,21)
_Fk = torch.exp(-_k2 / k0 ** 2)   # (21,21)
_Delta_k = (2 * pi / Lx) * (2 * pi / Ly)

# Parameter bounds (min, max) for scaling the network outputs
_param_bounds = torch.tensor([
    [0.0, 30.0],    # G_ee
    [-30.0, 0.0],   # G_ei
    [0.0, 10.0],    # G_ese
    [-10.0, 0.0],   # G_esre
    [-1.0, 0.0],    # G_srs
    [10.0, 100.0],  # alpha
    [100.0, 400.0], # beta
    [0.01, 0.2],    # t0
], dtype=torch.float32)


# -----------------------------------------------------------------------------
# CTM spectral model (vectorised, differentiable)
# -----------------------------------------------------------------------------

def compute_ctm_psd(params: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Compute CTM analytic power spectrum.

    Parameters
    ----------
    params : (N, 8) tensor of *physical* parameters in the same order as bounds.
    freqs  : (F,) tensor of frequencies in Hertz (must be 1-D).

    Returns
    -------
    psd : (N, F) tensor (real, float32)
    """
    device = params.device
    dtype = params.dtype

    # Unpack parameters (shape (N,1,1,1) for seamless broadcasting with omega (1,F,1,1))
    G_ee, G_ei, G_ese, G_esre, G_srs, alpha, beta, t0 = [
        p.view(-1, 1, 1, 1) for p in params.split(1, dim=1)
    ]

    # Pre-computed spatial grid moved to the same device / dtype
    k2 = _k2.to(device=device, dtype=dtype)           # (21,21)
    Fk = _Fk.to(device=device, dtype=dtype)

    omega = 2 * pi * freqs.to(device=device, dtype=dtype)  # (F,)
    omega = omega.view(1, -1, 1, 1)                        # (1,F,1,1)

    # Broadcast helpers
    N = params.shape[0]

    # L(omega)
    Lw = 1.0 / ((1.0 - 1j * omega / alpha) * (1.0 - 1j * omega / beta))  # (N,F,1,1) complex

    # q^2 r_e^2 term (real)
    num = (1.0 - 1j * omega / gamma_e) ** 2 - 1.0
    den = 1.0 - G_ei * Lw
    bracket = (
        Lw * G_ee
        + (Lw ** 2 * G_ese + Lw ** 3 * G_esre) * torch.exp(1j * omega * t0) / (1.0 - Lw ** 2 * G_srs)
    )
    q2 = (num - bracket / den).real  # (N,F,1,1)

    # Denominator for phi
    denom = (1.0 - G_srs * Lw ** 2) * (1.0 - G_ei * Lw) * (k2 * r_e ** 2 + q2 * r_e ** 2)

    phi = G_ese * torch.exp(1j * omega * t0 / 2.0) / denom  # (N,F,21,21)

    P = torch.sum(torch.abs(phi) ** 2 * Fk, dim=(-2, -1))  # (N,F)
    return (P * _Delta_k).real.float()



# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Simple MLP classifier that consumes flattened theta
# -----------------------------------------------------------------------------

# EEG-to-theta encoder
class EEGToTheta(nn.Module):
    """Predict one 8-dim θ vector *per channel* (19×)"""

    def __init__(self, in_len: int = 7680, out_dim: int = 8):
        super().__init__()
        # Operates on the per-channel time-series: (B, 19, 7680)
        self.net = nn.Sequential(
            nn.Linear(in_len, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor):  # x: (B, 19, 7680)
        return self.net(x)  # → (B, 19, 8)

# NEW: Per-channel CNN encoder (depth-wise convolutions)
class EEGToThetaCNN(nn.Module):
    """Per-channel θ estimator using lightweight depth-wise 1-D convolutions.

    Preserves the *one θ per electrode* property by setting ``groups=19`` so
    no information is mixed across channels.  It is intended as a drop-in
    replacement for :class:`EEGToTheta`.
    """

    def __init__(self, seg_len: int = 7680, out_dim: int = 8):
        super().__init__()
        # Depth-wise convs: each channel has its own filter bank (shared weights)
        self.conv1 = nn.Conv1d(19, 19 * 4, kernel_size=7, stride=2, padding=3, groups=19)
        self.bn1   = nn.BatchNorm1d(19 * 4)
        self.conv2 = nn.Conv1d(19 * 4, 19 * 8, kernel_size=7, stride=2, padding=3, groups=19)
        self.bn2   = nn.BatchNorm1d(19 * 8)

        # Global average-pool over time dimension → (B, 19*8)
        self.fc    = nn.Linear(19 * 8, 19 * out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor):  # x: (B, 19, 7680)
        z = F.relu(self.bn1(self.conv1(x)))          # (B, 19*4, 3840)
        z = F.relu(self.bn2(self.conv2(z)))          # (B, 19*8, 1920)
        z = z.mean(dim=-1)                           # global avg pool → (B, 19*8)
        theta_flat = self.fc(z)                      # (B, 19*out_dim)
        return theta_flat.view(-1, 19, self.out_dim)

# NEW: Shared-weight single-channel CNN. Processes each electrode independently but with identical filters.
class EEGToThetaCNNShared(nn.Module):
    """Shared-weight per-electrode CNN (each electrode goes through the same 1-D CNN).

    Steps:
    1. Reshape input to merge batch and electrode dims.
    2. Single-channel CNN extracts features.
    3. Global average-pool over time.
    4. Fully-connected layer to 8-D θ.
    5. Reshape back to (B,19,8).
    """

    def __init__(self, seg_len: int = 7680, out_dim: int = 8):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),  # time ↓×2
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),  # ↓×4
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.fc = nn.Linear(16, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor):  # x: (B, 19, 7680)
        B, C, T = x.shape
        x_flat = x.view(B * C, 1, T)           # (B*19,1,T)
        z = self.extract(x_flat)               # (B*19,16,T/4)
        z = z.mean(dim=-1)                     # global avg over time → (B*19,16)
        theta = self.fc(z)                     # (B*19,8)
        return theta.view(B, C, self.out_dim)  # (B,19,8)

# NEW: simple MLP that turns the 19×8 theta array into abnormal/normal logits
class ThetaClassifier(nn.Module):
    """Classify (normal vs abnormal) from per-channel θ parameters."""

    def __init__(self, in_dim: int = 19 * 8, num_classes: int = 2):
        super().__init__()
        # Stronger head: three hidden layers + dropout
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, theta: torch.Tensor):  # theta: (B, 19, 8)
        flat = theta.view(theta.size(0), -1)  # (B, 152)
        return self.net(flat)  # (B, num_classes)






# -----------------------------------------------------------------------------
# Theta-only optimisation (Option A)
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# Joint optimisation: theta (per-sample) + classifier head
# -----------------------------------------------------------------------------

def train_theta_joint(
    data_root: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-2,
    w_psd: float = 1.0,
    ce_start: float = 0.05,
    ce_end: float = 0.5,
    ce_span: int = 50,
    save_theta: str = "theta_fitted.pt",
):
    """Optimise per-window theta by matching CTM PSD to empirical PSD."""
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS; CTM PSD will run on CPU if complex ops unsupported.")
    else:
        device = torch.device("cpu")

    # Data
    train_dl, eval_dl, train_ds, eval_ds = build_split_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=4,
        random_segment_train=False,
        random_segment_eval=False,
    )
    print(f"Train size: {len(train_ds)} | Eval size: {len(eval_ds) if eval_ds is not None else 0}")
    N = len(train_ds)
    p_dim = 8  # number of CTM parameters per window (no channel dimension)

    # Encoder
    # Choose encoder architecture
    # encoder = EEGToTheta(in_len=7680, out_dim=p_dim).to(device)            # simple MLP
    # encoder = EEGToThetaCNN(seg_len=7680, out_dim=p_dim).to(device)        # depth-wise CNN (separate weights per channel)
    encoder = EEGToThetaCNNShared(seg_len=7680, out_dim=p_dim).to(device)    # shared-weight CNN
    classifier = ThetaClassifier().to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)

    from scipy.signal import welch
    def compute_empirical_psd(x: torch.Tensor, freqs: torch.Tensor, fs=128, n_fft=512) -> torch.Tensor:
        """Return empirical PSD per channel → (B, 19, F)."""
        from scipy.interpolate import interp1d
        B = x.shape[0]
        psds = []
        target_freqs = freqs.cpu().numpy()
        for i in range(B):
            p_ch = []
            for ch in range(19):
                f, Pxx = welch(x[i, ch].cpu().numpy(), fs=fs, nperseg=n_fft)
                interp_func = interp1d(f, Pxx, kind='linear', bounds_error=False, fill_value='extrapolate')
                p_ch.append(interp_func(target_freqs))  # (F,)
            psds.append(np.stack(p_ch))  # (19,F)
        return torch.tensor(np.stack(psds), dtype=torch.float32)  # (B,19,F)

    freqs = torch.tensor(np.linspace(0.5, 40, 100), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Loss helper – same weighting scheme as archive/pipeline/.../cortico_thalamic.py
    # ------------------------------------------------------------------
    def weighted_log_mse(pred_psd: torch.Tensor, target_psd: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Replicates _loss_function() from cortico_thalamic but in PyTorch.

        Emphasises the alpha band (8–12 Hz) by weighting its squared error 5×.
        Expects shape (..., F) for *pred_psd* and *target_psd* and uses the
        global *freqs* tensor defined above (1-D, length F).
        """

        log_pred = torch.log10(pred_psd + eps)
        log_tgt  = torch.log10(target_psd + eps)

        diff2 = (log_pred - log_tgt) ** 2  # (..., F)

        w = torch.ones_like(freqs, device=pred_psd.device)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        w[alpha_mask] = 5.0

        return (diff2 * w.view(*(1,) * (diff2.dim() - 1), -1)).mean()

    # ----------------------------------------------------------
    # Select a fixed sample for diagnostic plotting (first item)
    # ----------------------------------------------------------
    sample_idx = 1
    x_sample, _, _ = train_ds[sample_idx]
    x_sample = x_sample.unsqueeze(0).to(device)  # (1, 19, 7680)

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        # CE loss weight scheduler: linear ramp ce_start → ce_end over ce_span epochs
        progress = min(epoch - 1, ce_span - 1) / (ce_span - 1) if ce_span > 1 else 1.0
        ce_weight = ce_start + (ce_end - ce_start) * progress
        running_psd = running_ce = 0.0
        correct = 0
        val_running_psd = val_running_ce = val_correct = 0
        for x, y, idx in train_dl:
            x = x.to(device)
            y = y.to(device)

            theta_raw = encoder(x)  # (B,19,8)
            # Map to physical parameter bounds
            min_b = _param_bounds[:, 0].to(device)
            max_b = _param_bounds[:, 1].to(device)
            theta_clamped = torch.sigmoid(theta_raw)
            theta_phys = min_b + theta_clamped * (max_b - min_b)  # (B,19,8)

            B_cur = theta_phys.size(0)
            pred_psd_flat = compute_ctm_psd(theta_phys.view(-1, theta_phys.size(-1)), freqs).to(device)
            pred_psd = pred_psd_flat.view(B_cur, 19, -1)  # (B,19,F)
            
            target_psd = compute_empirical_psd(x.cpu(), freqs).to(device)  # (B,19,F)
            
            psd_loss = weighted_log_mse(pred_psd, target_psd)
 
            # Classification head
            logits = classifier(theta_phys)  # (B,2)
            ce_loss = F.cross_entropy(logits, y)
            
            # Apply current CE weight from scheduler
            total_loss = w_psd * psd_loss + ce_weight * ce_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            running_psd += psd_loss.item() * x.size(0)
            running_ce  += ce_loss.item()  * x.size(0)
            correct     += (logits.argmax(dim=1) == y).sum().item()

       

        # -------------------------
        # Evaluation (no gradient)
        # -------------------------
        if eval_dl is not None:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                for x_v, y_v, _ in eval_dl:
                    x_v = x_v.to(device)
                    y_v = y_v.to(device)
                    theta_raw_v = encoder(x_v)
                    min_b_v = _param_bounds[:, 0].to(device)
                    max_b_v = _param_bounds[:, 1].to(device)
                    theta_phys_v = min_b_v + torch.sigmoid(theta_raw_v) * (max_b_v - min_b_v)

                    Bv = theta_phys_v.size(0)
                    pred_psd_v = compute_ctm_psd(theta_phys_v.view(-1, theta_phys_v.size(-1)), freqs).to(device)
                    pred_psd_v = pred_psd_v.view(Bv, 19, -1)
                    target_psd_v = compute_empirical_psd(x_v.cpu(), freqs).to(device)

                    psd_val = weighted_log_mse(pred_psd_v, target_psd_v)

                    logits_v = classifier(theta_phys_v)
                    ce_val   = F.cross_entropy(logits_v, y_v)

                    val_running_psd += psd_val.item() * x_v.size(0)
                    val_running_ce  += ce_val.item()  * x_v.size(0)
                    val_correct     += (logits_v.argmax(dim=1) == y_v).sum().item()
        train_acc = correct / len(train_ds)
        if eval_ds is not None and len(eval_ds) > 0:
            val_psd_mean = val_running_psd / len(eval_ds)
            val_ce_mean  = val_running_ce  / len(eval_ds)
            val_acc      = val_correct     / len(eval_ds)
        else:
            val_psd_mean = val_ce_mean = val_acc = 0.0
        beta_curr = ce_weight
        print(
            f"Epoch {epoch}: PSD={running_psd/len(train_ds):.4f}, CE={running_ce/len(train_ds):.4f}, Acc={train_acc:.3f} | "
            f"Val PSD={val_psd_mean:.4f}, Val CE={val_ce_mean:.4f}, Val Acc={val_acc:.3f} | w_PSD={w_psd:.2f} | w_CE={beta_curr:.2f}"
        )

        # ---------------------------------------
        # Diagnostic plot every 5 epochs
        # ---------------------------------------
        if epoch % 5 == 0:
            encoder.eval()
            with torch.no_grad():
                theta_s = encoder(x_sample)  # (1,19,8)
                pred_psd_s = compute_ctm_psd(theta_s.view(-1, theta_s.size(-1)), freqs).cpu().numpy().reshape(1,19,-1)
                target_psd_s = compute_empirical_psd(x_sample.cpu(), freqs).cpu().numpy()  # (1,19,F)

            # Average across channels for plotting clarity
            pred_psd_avg = pred_psd_s.mean(axis=1)[0]  # (F,)
            target_psd_avg = target_psd_s.mean(axis=1)[0]

            # Convert to decibels for better visual comparison
            eps = 1e-12
            plt.figure(figsize=(6, 4))
            plt.plot(freqs.cpu().numpy(), 10*np.log10(target_psd_avg + eps), label="Empirical", color="black")
            plt.plot(freqs.cpu().numpy(), 10*np.log10(pred_psd_avg + eps), label="CTM fit", color="red", linestyle="--")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (dB)")
            plt.title(f"PSD fit (epoch {epoch})")
            plt.legend()
            import os
            os.makedirs("plots", exist_ok=True)
            out_fname = f"plots/fit_epoch_{epoch}.png"
            plt.tight_layout()
            plt.savefig(out_fname)
            plt.close()
            #print(f"           saved diagnostic plot to {out_fname}")

            # --- Shape-only plot (magnitude removed) ---
            log_pred_shape = 10 * np.log10(pred_psd_s + eps)
            log_tgt_shape  = 10 * np.log10(target_psd_s + eps)
            # Zero-mean across frequency (shape only)
            log_pred_shape = log_pred_shape - log_pred_shape.mean(axis=-1, keepdims=True)
            log_tgt_shape  = log_tgt_shape - log_tgt_shape.mean(axis=-1, keepdims=True)

            # Average across channels for plotting
            log_pred_shape_avg = log_pred_shape.mean(axis=1)[0]  # (F,)
            log_tgt_shape_avg  = log_tgt_shape.mean(axis=1)[0]

            plt.figure(figsize=(6, 4))
            plt.plot(freqs.cpu().numpy(), log_tgt_shape_avg, label="Empirical (shape)", color="black")
            plt.plot(freqs.cpu().numpy(), log_pred_shape_avg, label="CTM fit (shape)", color="red", linestyle="--")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Relative log-PSD (dB, zero-mean)")
            plt.title(f"PSD shape comparison (epoch {epoch})")
            plt.legend()
            out_shape = f"plots/fit_shape_epoch_{epoch}.png"
            plt.tight_layout()
            plt.savefig(out_shape)
            plt.close()
            print(f"           saved shape-only plot to {out_shape}")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint optimisation of theta (CTM)")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root. If it contains 'train/' and 'eval/' sub-folders they will be used as splits; otherwise the folder itself is treated as a single dataset.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_theta", type=str, default="theta_fitted.pt", help="Path to save learned theta")
    parser.add_argument("--w_psd", type=float, default=1, help="Weight for PSD loss term")
    parser.add_argument("--ce_start", type=float, default=0.1, help="Initial CE loss weight")
    parser.add_argument("--ce_end", type=float, default=0.1, help="Final CE loss weight after ce_span epochs")
    parser.add_argument("--ce_span", type=int, default=50, help="Number of epochs over which CE weight is linearly increased")

    args = parser.parse_args()
    train_theta_joint(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_psd=args.w_psd,
        ce_start=args.ce_start,
        ce_end=args.ce_end,
        ce_span=args.ce_span,
        save_theta=args.save_theta,
    )
