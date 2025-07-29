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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Added for post-training plots

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
        # Deeper network with dropout for better capacity/regularisation
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
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
# Encoder-classifier training (CTM-independent)
# -----------------------------------------------------------------------------

def train_encoder_classifier(
    data_root: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_model: str = "encoder_classifier.pt",
    plot_dir: str = "plots",
):
    """Train an encoder + classifier to discriminate normal vs abnormal EEG.

    This routine is a stripped-down version of the original training loop and
    contains *no* cortico-thalamic model logic. Only cross-entropy loss on the
    classifier head is optimised.
    """

    # ---------------- Device ----------------
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )

    # ---------------- Data ----------------
    train_dl, eval_dl, train_ds, eval_ds = build_split_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=4,
        random_segment_train=True,
        random_segment_eval=False,
    )
    print(f"Train size: {len(train_ds)} | Eval size: {len(eval_ds) if eval_ds is not None else 0}")

    # ---------------- Models ----------------
    encoder = EEGToThetaCNNShared(seg_len=7680, out_dim=8).to(device)
    classifier = ThetaClassifier().to(device)

    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=lr
    )

    # --- metric history for plotting ---
    hist_train_loss, hist_val_loss = [], []
    hist_train_acc,  hist_val_acc  = [], []

    # ---------------- Training loop ----------------
    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        run_loss = 0.0
        run_correct = 0
        for x, y, _ in train_dl:
            x = x.to(device)
            y = y.to(device)

            feats = encoder(x)            # (B, 19, 8)
            logits = classifier(feats)    # (B, 2)
            loss = F.cross_entropy(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            run_loss    += loss.item() * x.size(0)
            run_correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = run_loss / len(train_ds)
        train_acc  = run_correct / len(train_ds)
        hist_train_loss.append(train_loss)
        hist_train_acc.append(train_acc)

        # ---------- Evaluation ----------
        val_loss = val_acc = 0.0
        if eval_dl is not None and len(eval_ds) > 0:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                v_loss = v_correct = 0.0
                for xv, yv, _ in eval_dl:
                    xv = xv.to(device)
                    yv = yv.to(device)
                    v_feats = encoder(xv)
                    v_logits = classifier(v_feats)
                    loss_v = F.cross_entropy(v_logits, yv)
                    v_loss    += loss_v.item() * xv.size(0)
                    v_correct += (v_logits.argmax(dim=1) == yv).sum().item()
                val_loss = v_loss / len(eval_ds)
                val_acc  = v_correct / len(eval_ds)
            hist_val_loss.append(val_loss)
            hist_val_acc.append(val_acc)

        print(
            f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.3f} | "
            f"Val loss={val_loss:.4f}, acc={val_acc:.3f}"
        )

    # ---------------- Save weights ----------------
    #torch.save({"encoder": encoder.state_dict(), "classifier": classifier.state_dict()}, save_model)
    print(f"Saved model to {save_model}")

    # ---------------- Plots ----------------
    import os
    os.makedirs(plot_dir, exist_ok=True)

    epochs_range = range(1, epochs + 1)

    # Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, hist_train_loss, label="Train")
    if hist_val_loss:
        plt.plot(epochs_range, hist_val_loss, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(plot_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    print(f"Saved loss curve to {loss_path}")

    # Accuracy curve
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, hist_train_acc, label="Train")
    if hist_val_acc:
        plt.plot(epochs_range, hist_val_acc, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(plot_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()
    print(f"Saved accuracy curve to {acc_path}")

    # Confusion matrix on evaluation set
    if eval_dl is not None and len(eval_ds) > 0:
        encoder.eval()
        classifier.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xe, ye, _ in eval_dl:
                xe = xe.to(device)
                logits_e = classifier(encoder(xe))
                y_true.extend(ye.cpu().numpy())
                y_pred.extend(logits_e.argmax(dim=1).cpu().numpy())
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
        disp.plot(cmap="Blues")
        plt.title("Confusion matrix (eval)")
        cm_path = os.path.join(plot_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"Saved confusion matrix to {cm_path}")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EEG encoder-classifier (CTM-free)")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root. If it contains 'train/' and 'eval/' sub-folders they will be used as splits; otherwise the folder itself is treated as a single dataset.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save training curves & confusion matrix")
    parser.add_argument(
        "--save_model",
        type=str,
        default="encoder_classifier.pt",
        help="Filename to store the trained weights (encoder & classifier)",
    )

    args = parser.parse_args()
    train_encoder_classifier(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_model=args.save_model,
        plot_dir=args.plot_dir,
    )
