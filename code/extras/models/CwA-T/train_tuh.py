#!/usr/bin/env python
# coding: utf-8

"""
Training script for the cleaned TUH Abnormal/Normal EEG dataset (60-second segments at 128 Hz).
It mirrors the logic of `train.py`, but:
  • Loads MNE `.fif` files written by `cleanup_real_eeg_tuh.py`
  • Infers the label from the parent directory (`normal` ⇢ 0, `abnormal` ⇢ 1); no CSV is required.
  • Uses the same model architecture, optimisation, logging and evaluation pipeline as `train.py`.

Example usage
-------------
$ python code/extras/models/CwA-T/train_tuh.py code/extras/models/CwA-T/configs/encoderS+transformer.yml
"""

# Standard lib
import os
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import math

# Third-party
import yaml
import mne
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# ——— Local imports (same as train.py) ———————————————————————————————
from models.encoder import res_encoderS, res_encoderM, res_encoderL
try:
    from data_preprocessing.data_loading import load_data as load_tuh_data
except ModuleNotFoundError:
    # fallback when script is executed from project root
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))  # add project root to PYTHONPATH
    from data_preprocessing.data_loading import load_data as load_tuh_data
from models.classifier import transformer_classifier

# ————————————————————————————————————————————————————————————————
# Utility
# ————————————————————————————————————————————————————————————————

def transform(data: Tensor) -> Tensor:
    """Per-segment channel-wise z-score normalisation.

    For each channel (column) compute mean & std across the 60-s segment and
    normalise (x-μ)/σ.  Adds a small epsilon to avoid division by zero.
    """
    eps = 1e-8
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, unbiased=False, keepdim=True) + eps
    return (data - mean) / std

# ————————————————————————————————————————————————————————————————
# Dataset
# ————————————————————————————————————————————————————————————————

class RawListDataset(Dataset):
    """Dataset that wraps list of (mne.Raw, sex, age, label).

    Applies resampling → channel picking → normalisation, returning tensor + label.
    """

    def __init__(self, raw_list: List[tuple], n_channels: int, target_sfreq: int, transform=None):
        self.raw_list = raw_list
        self.n_channels = n_channels
        self.target_sfreq = target_sfreq
        self.segment_len = target_sfreq * 60
        self.transform = transform

        # canonical channel list as before
        self.canonical = [
            "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
        ]

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, idx):
        #print(f"[DEBUG] Fetching item {idx}")
        raw, _, _, label = self.raw_list[idx]

        if abs(raw.info['sfreq'] - self.target_sfreq) > 1e-3:
            raw = raw.copy().resample(self.target_sfreq, npad="auto", verbose=False)

        # pick channels
        picks = []
        for ch in self.canonical:
            for v in [ch, ch.upper(), ch.capitalize(), f"EEG {ch.upper()}-REF", f"EEG {ch.upper()}-LE", f"EEG {ch.upper()}-CAR"]:
                if v in raw.ch_names:
                    picks.append(v)
                    break
        data_np = raw.copy().pick_channels(picks).get_data()
        tensor = torch.from_numpy(data_np.astype(np.float32).T)

        if tensor.shape[0] > self.segment_len:
            tensor = tensor[: self.segment_len, :]
        elif tensor.shape[0] < self.segment_len:
            pad = self.segment_len - tensor.shape[0]
            tensor = torch.cat([tensor, torch.zeros(pad, tensor.shape[1])], dim=0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label), "raw"

# ————————————————————————————————————————————————————————————————
# Model wrapper (identical to train.py)
# ————————————————————————————————————————————————————————————————

class Model(nn.Module):
    def __init__(self, input_size: int, n_channels: int, model_hyp: dict, n_classes: int):
        super().__init__()
        self.ae = res_encoderL(n_channels=n_channels, groups=n_channels, num_classes=n_classes, d_model=model_hyp["d_model"])
        self.dropout = nn.Identity() if model_hyp.get("dropout", 0.0) == 0 else nn.Dropout(model_hyp.get("dropout", 0.0))
        if model_hyp.get("classifier", "transformer") == "mlp_1l":
            from models.classifier import MLP_1l
            self.transformer_encoder = MLP_1l(n_channels, model_hyp["d_model"], n_classes)
        elif model_hyp.get("classifier", "transformer") == "mlp_3l":
            from models.classifier import MLP_3l
            self.transformer_encoder = MLP_3l(n_channels, model_hyp["d_model"], n_classes)
        else:
            self.transformer_encoder = transformer_classifier(input_size, n_channels, model_hyp, n_classes)
        self.reset_parameters()

    # ------------------------------------------------------------- #
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logging.info("Parameters initialised.")

    # ------------------------------------------------------------- #
    def forward(self, x):
        z = x.transpose(-1, -2)      # (B, C, T)
        z = self.ae(z)
        z = self.dropout(z)
        y = self.transformer_encoder(z)
        return y

# ————————————————————————————————————————————————————————————————
# Scheduler & evaluation helpers (copied from train.py)
# ————————————————————————————————————————————————————————————————

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=0, power=0.9):
    if max_iter == 0:
        raise ValueError("max_iter cannot be zero!")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer.param_groups[0]["lr"]

# ------------------------------------------------------------- #

# Overall accuracy evaluation (per-segment)

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Return overall accuracy along with correct and total sample counts."""
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for signals, labels, _ in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total

# ————————————————————————————————————————————————————————————————
# Training entry-point
# ————————————————————————————————————————————————————————————————

def train(cfg: dict):
    print("[DEBUG] Entered train()")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"].get("use_cuda", 1) else "cpu")

    # Load raw lists using data_loading style
    n_ch = cfg["n_channels"]
    target_sfreq = cfg.get("processing", {}).get("frequency", 100)

    train_raw_list = load_tuh_data(cfg["dataset"]["train_data_dir"])
    val_raw_list = load_tuh_data(cfg["dataset"]["val_data_dir"])
    print("[DEBUG] Loaded train and validation raw lists")

    train_set = RawListDataset(train_raw_list, n_channels=n_ch, target_sfreq=target_sfreq, transform=transform)
    val_set = RawListDataset(val_raw_list, n_channels=n_ch, target_sfreq=target_sfreq, transform=transform)
    print("[DEBUG] Created RawListDataset instances")

    # Debug tiny-overfit mode: use only first N samples to check if model can overfit
    debug_n = cfg.get("debug_overfit", 0)
    if isinstance(debug_n, int) and debug_n > 0:
        train_set = Subset(train_set, list(range(min(debug_n, len(train_set)))))
        val_set = Subset(val_set, list(range(min(debug_n, len(val_set)))))  # keep small val for speed
        logging.warning(f"Debug overfit mode ENABLED: using first {min(debug_n, len(train_set))} train and {min(debug_n, len(val_set))} val samples")

    train_loader = DataLoader(train_set,
                              batch_size=cfg["train"]["batch_size"],
                              shuffle=cfg["dataset"].get("shuffle", True),
                              num_workers=cfg["dataset"].get("num_workers", 4),
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=cfg["train"].get("batch_size", 1),
                            shuffle=False,
                            num_workers= cfg["dataset"].get("num_workers", 4),
                            pin_memory=True)
    print("[DEBUG] DataLoaders initialized")

    # Model
    model_name = cfg["model"]["name"]
    net = Model(cfg["input_size"], cfg["n_channels"], cfg["model"], len(cfg["dataset"]["classes"])).to(device)
    print("[DEBUG] Model initialized")

    # Optionally resume / load weights
    ckpt_cfg = cfg["checkpoint"]
    weight_path = ckpt_cfg.get("weights")
    if weight_path and Path(weight_path).is_file():
        logging.info(f"Loading pre-trained weights from {weight_path}")
        net.load_state_dict(torch.load(weight_path, map_location=device))

    # Optimiser & criterion
    opt_cfg = cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adam").lower()
    lr = float(opt_cfg.get("init_lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if opt_name == "adamw":
        # Exclude biases and norm layers from weight decay
        no_decay_keys = ["bias", "LayerNorm.weight", "layer_norm.weight", "bn.weight", "batchnorm.weight"]
        decay_params, no_decay_params = [], []
        for n, p in net.named_parameters():
            if any(k in n for k in no_decay_keys):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        optim = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", False))
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    else:  # default Adam
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    # ---- Scheduler wiring from config ----
    sched_cfg = cfg.get("scheduler", {})
    use_warmup = cfg.get("warmup", 0) == 1
    warmup_steps = cfg.get("train", {}).get("warmup_steps", 0)

    steps_per_epoch = len(train_loader)
    epochs = cfg["train"]["n_epochs"]
    total_steps = epochs * steps_per_epoch
    lr_init = float(cfg["optimizer"]["init_lr"])
    lr_min = float(sched_cfg.get("lr_min", 0.0))

    scheduler = None
    if sched_cfg.get("name", "").lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=lr_min)
    elif sched_cfg.get("name", "").lower() == "cosine_restart":
        T_0 = sched_cfg.get("T_0", 10)
        T_mult = sched_cfg.get("T_mult", 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
    # else: leave scheduler as None (no scheduling)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("criterion", {}).get("label_smoothing", 0.0))

    print("[DEBUG] Optimizer and scheduler set up")

    # TensorBoard
    tb_dir = Path(cfg["tensorboard"]["runs_dir"]) / f"{datetime.now().strftime('%y%m%d%H%M')}_{model_name}_tuh_board"
    writer = SummaryWriter(str(tb_dir))

    # Training loop
    epochs = cfg["train"]["n_epochs"]
    global_step = 0

    best_acc = 0.0  # tracked for info only; not used for saving
    start_time = time.time()
    patience_cfg = cfg["train"].get("early_stopping", {})
    patience = patience_cfg.get("patience", 0)
    if patience > 0:
        wait = 0

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        print(f"[DEBUG] Getting first batch from train_loader")
        for batch_idx, (signals, labels, _) in enumerate(train_loader):
            optim.zero_grad()
            signals, labels = signals.to(device), labels.to(device)
            outputs = net(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
            # Move scheduler stepping outside the batch loop for epoch-based schedulers
            # log lr occasionally
            if scheduler is not None and global_step % 50 == 0:
                writer.add_scalar("train/lr", optim.param_groups[0]["lr"], global_step)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Step scheduler per epoch for epoch-based schedulers
        if scheduler is not None and sched_cfg.get("name", "").lower() in {"cosine_restart"}:
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Evaluation after each epoch
        acc, correct, total = evaluate_model(net, val_loader, device)
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss {epoch_loss:.4f} | Train Acc {train_acc:.4f} | Val Acc {acc:.4f} ({correct}/{total})")
        writer.add_hparams({"epoch": epoch+1}, {"train_loss": epoch_loss, "train_acc": train_acc, "val_acc": acc}, run_name=f"epoch_{epoch+1}")

        # Checkpoint saving disabled as per user request
        if acc > best_acc:
            best_acc = acc
            #logging.info(f"New best accuracy {best_acc:.4f} (no weights saved)")
            wait = 0  # reset patience counter on improvement
        else:
            if patience > 0:
                wait += 1
                if wait >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1} (no val improvement for {patience} epochs)")
                    break

    # ----------------------------------------------------------------- #
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    logging.info(f"Training finished in {hours}h {minutes}m {seconds}s")
    writer.close()
    return best_acc

# ————————————————————————————————————————————————————————————————
# Main
# ————————————————————————————————————————————————————————————————

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CwA-T on cleaned TUH EEG abnormality dataset")
    parser.add_argument("config_file", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Configuration
    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Print configuration parameters once
    print("===== Training Configuration =====")
    print(yaml.dump(cfg, sort_keys=False))
    print("===================================")

    # Logging setup
    log_dir = Path("../logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_name = cfg["model"]["name"]
    log_file = log_dir / f"{datetime.now().strftime('%y%m%d%H%M')}_{model_name}_tuh.log"
    logging.basicConfig(level=logging.INFO, filename=str(log_file), filemode="w", format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to stdout

    train(cfg)
