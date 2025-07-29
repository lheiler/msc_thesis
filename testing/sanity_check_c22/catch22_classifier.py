# Catch22 feature-based classifier for TUH EEG abnormal / normal
# Author: AI assistant
#
# This script reuses the EEGDataset from prototype_ctm_coupled.py to load either
# the NumPy-blob or folder-based TUH dataset. For every 60-second window it
# computes the 22 catch22 features for each of the 19 channels (total 418
# features) and trains a lightweight MLP classifier.
#
# Usage
# -----
#   python catch22_classifier.py --data_root /path/to/tuh-eeg-ab-clean \
#          --epochs 20 --batch_size 32 --lr 1e-3
#
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pycatch22 import catch22_all  # pip install pycatch22

# Reuse the dataset definition from the prototype script (safe because guarded
# by if __name__ == "__main__" in that file)
from prototype_ctm_coupled import EEGDataset 


# -----------------------------------------------------------------------------
# Multiprocessing worker (must be at module level for pickling on macOS/Windows)
# -----------------------------------------------------------------------------


def row_worker(payload):
    """Compute catch22 features for a single EEG window.

    Parameters
    ----------
    payload : tuple(ndarray, int)
        A tuple containing the 19×T numpy array and its integer label.

    Returns
    -------
    list
        418 features followed by the label.
    """
    arr, label = payload
    feats = extract_catch22_features(arr)
    return feats.tolist() + [label]


# -----------------------------------------------------------------------------
# Feature-extraction wrapper
# -----------------------------------------------------------------------------


def extract_catch22_features(window: np.ndarray) -> np.ndarray:
    """Compute 22 catch22 features for each of the 19 EEG channels.

    Parameters
    ----------
    window : ndarray, shape (19, T)

    Returns
    -------
    ndarray, shape (418,)
    """
    feats = []
    for ch in range(window.shape[0]):
        feats.extend(catch22_all(window[ch])["values"])
    return np.asarray(feats, dtype=np.float32)


# -----------------------------------------------------------------------------
# CSV feature dataset (pre-computed)
# -----------------------------------------------------------------------------


# ----------------------------- CSV dataset -----------------------------

class CSVFeatureDataset(Dataset):
    """Load feature matrix and labels from a CSV produced by this script."""

    def __init__(self, csv_path: str):
        import numpy as np
        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.features = arr[:, :-1].astype(np.float32)
        self.labels = arr[:, -1].astype(np.int64)

    def __len__(self) -> int:  # noqa: D401
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 418, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,in_dim)
        return self.net(x)


# -----------------------------------------------------------------------------
# Training / evaluation helpers
# -----------------------------------------------------------------------------


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
    return correct / len(loader.dataset)


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def train(
    data_root: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
):
    # Device (CPU is fine because features computed on CPU; classifier small)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- dataset paths ----------
    train_dir = os.path.join(data_root, "train")
    eval_dir = os.path.join(data_root, "eval")

    csv_train = os.path.abspath("catch22_features_train.csv")
    csv_eval = os.path.abspath("catch22_features_eval.csv")

    # ---------- compute & save features if CSV missing ----------

    def compute_and_save(split_path: str, csv_path: str, random_segment: bool):
        if os.path.exists(csv_path):
            return  # already there

        print(f"Computing catch22 features for {split_path} → {csv_path}")
        base_ds = EEGDataset(split_path, random_segment=random_segment)

        import csv
        header = [f"feat_{i}" for i in range(418)] + ["label"]
        from multiprocessing import Pool, cpu_count

        iterable = [
            (base_ds[idx][0].numpy(), int(base_ds[idx][1].item())) for idx in range(len(base_ds))
        ]

        with Pool(processes=cpu_count()) as pool, open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in tqdm(pool.imap_unordered(row_worker, iterable, chunksize=4), total=len(base_ds), desc="extract"):
                writer.writerow(row)

    # Train split
    if os.path.isdir(train_dir):
        compute_and_save(train_dir, csv_train, random_segment=True)
    else:
        # dataset without explicit split
        compute_and_save(data_root, csv_train, random_segment=True)

    # Eval split (optional)
    if os.path.isdir(eval_dir):
        compute_and_save(eval_dir, csv_eval, random_segment=False)

    # ---------- load CSV datasets ----------
    ds_train = CSVFeatureDataset(csv_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)

    ds_eval = dl_eval = None
    if os.path.exists(csv_eval):
        ds_eval = CSVFeatureDataset(csv_eval)
        dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=4)

    n_classes = int(ds_train.labels.max() + 1)

    # ---------- model ----------
    model = MLPClassifier(n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    # ---------- training ----------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce_loss(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * xb.size(0)

        print(f"Epoch {epoch}: train loss = {running / len(ds_train):.5f}")

        # Optional eval each epoch
        if dl_eval is not None:
            acc_eval = accuracy(model, dl_eval, device)
            print(f"            eval acc = {acc_eval * 100:.2f}%")

    # Final metrics
    acc_train = accuracy(model, dl_train, device)
    print(f"Final train accuracy: {acc_train * 100:.2f}%")
    if dl_eval is not None:
        acc_eval = accuracy(model, dl_eval, device)
        print(f"Final eval accuracy:  {acc_eval * 100:.2f}%")

    # ----- save model -----
    save_path = os.path.join(data_root, "catch22_classifier.pth")  # model still inside dataset root
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # features already saved before training; no need to export here


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Catch22 feature-based EEG classifier")
    p.add_argument("--data_root", required=True, type=str, help="Path to TUH dataset root")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()
    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    ) 