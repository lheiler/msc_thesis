import tqdm
from pathlib import Path
import sys, os, time, logging, argparse, yaml, math, torch, shutil
import multiprocessing
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import nn, Tensor
from datetime import datetime
from typing import List, Tuple
from collections import OrderedDict
import mne
mne.set_log_level("WARNING")  
# -----------------------------------------------------------------------------
#  Make code/ directory importable so we can access the shared preprocessing util
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve()
CODE_ROOT = THIS_DIR.parents[3]  # .../thesis/code
sys.path.append(str(CODE_ROOT / "data_preprocessing"))

from data_loading import load_data  # noqa: E402

# -----------------------------------------------------------------------------
#  Model imports (same as original CwA-T train.py)
# -----------------------------------------------------------------------------
# Import all encoder variants so we can choose at runtime
from models.encoder import res_encoderS, res_encoderM, res_encoderL  # noqa: E402
from models.classifier import transformer_classifier  # noqa: E402

logger = logging.getLogger("train_tuh")
logger.setLevel(logging.INFO)

# Add automatic device selection ------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
#  Simple normalisation helper (identical to original train.py)
# -----------------------------------------------------------------------------

def z_normalise(data: Tensor, mean: Tensor, std: Tensor):
    """Apply channel-wise z-normalisation."""
    return (data - mean) / std

# -----------------------------------------------------------------------------
#  Custom Dataset for TUH EEG .fif files
# -----------------------------------------------------------------------------

class TUHDataset(Dataset):
    """Wraps the (raw, sex_code, abn) tuples returned by load_data().

    The raw signal is converted to a float32 torch tensor of shape
    (seq_len, n_channels) to match the original CwA-T pipeline. The dataset
    does all on-the-fly cropping/resampling so we avoid storing temporary
    files on disk.
    """

    def __init__(
        self,
        tuples: List[Tuple["mne.io.Raw", int, int]],
        *,
        target_len: int,
        target_sf: int,
        n_channels: int,
        mean: List[float] | None = None,
        std: List[float] | None = None,
    ) -> None:
        super().__init__()
        self.samples = tuples
        self.target_len = target_len
        self.target_sf = target_sf
        self.n_channels = n_channels
        self.mean = torch.tensor(mean) if mean is not None else None
        self.std = torch.tensor(std) if std is not None else None
        # Cache processed signals so we do not resample/crop every epoch.
        self._cache: dict[int, tuple[Tensor, Tensor, str]] = {}

    def __len__(self):
        return len(self.samples)

    def _process_raw(self, raw):
        """Resample & crop raw to desired length, return np array (T, C)."""
        # Resample if needed
        if int(raw.info["sfreq"]) != self.target_sf:
            raw = raw.copy().resample(self.target_sf)

        data = raw.get_data(picks="eeg")  # (C, T)
        # Ensure we only use the first n_channels
        data = data[: self.n_channels, :]

        # Crop/pad to target length (seconds * sf)
        if data.shape[1] >= self.target_len:
            data = data[:, : self.target_len]
        else:
            pad_width = self.target_len - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

        # to (T, C)
        return torch.tensor(data.T, dtype=torch.float32)

    def __getitem__(self, idx):
        # Return cached version if available in this worker
        if idx in self._cache:
            return self._cache[idx]

        #print(f"[DEBUG] Fetching item {self.samples[idx]}")
        raw, _sex_code, _, abn = self.samples[idx]
        signal = self._process_raw(raw)

        if self.mean is not None and self.std is not None:
            signal = z_normalise(signal, self.mean, self.std)

        # Build a stable file identifier for per-case evaluation (filename stem)
        fname = os.path.basename(raw.filenames[0]) if raw.filenames else f"sample_{idx}.fif"

        sample = (signal, torch.tensor(abn, dtype=torch.long), fname)
        self._cache[idx] = sample
        return sample

# -----------------------------------------------------------------------------
#  Model wrapper (identical to original but renamed for clarity)
# -----------------------------------------------------------------------------

class CwATModel(nn.Module):
    """Wrapper that selects the Auto-Encoder depth (S/M/L) based on the config."""

    def __init__(
        self,
        *,
        input_size: int,
        n_channels: int,
        hyp: dict,
        num_classes: int,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Select encoder variant
        # ------------------------------------------------------------------
        encoder_name: str = hyp.get("name", "encoderM")  # fallback if not provided
        if "L" in encoder_name.upper():
            encoder_fn = res_encoderL
            print("Using encoderL")
        elif "S" in encoder_name.upper():
            encoder_fn = res_encoderS
            print("Using encoderS")
        else:
            encoder_fn = res_encoderM  # default
            print("Using encoderM")

        self.encoder = encoder_fn(
            n_channels=n_channels,
            groups=n_channels,
            num_classes=num_classes,
            d_model=hyp["d_model"],
        )

        # Transformer head stays unchanged
        self.transformer_head = transformer_classifier(input_size, n_channels, hyp, num_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor):
        # Original pipeline expects (B, T, C). Encoder needs (B, C, T)
        z = self.encoder(x.transpose(-1, -2))
        return self.transformer_head(z)

# -----------------------------------------------------------------------------
#  Training / Evaluation helpers (refactored from original train.py)
# -----------------------------------------------------------------------------

def poly_lr_scheduler(optimizer, init_lr, cur_iter, *, max_iter, power=0.9):
    lr = init_lr * (1 - cur_iter / max_iter) ** power
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def accuracy(output: Tensor, target: Tensor):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct / target.size(0)


# -----------------------------------------------------------------------------
#  Main training loop
# -----------------------------------------------------------------------------

def train(cfg: dict):
    # Load data ----------------------------------------------------------------
    logger.info("Loading TUH EEG dataset ...")
    train_samples = load_data(cfg["dataset"]["train_data_dir"])
    val_samples = load_data(cfg["dataset"]["val_data_dir"])

    input_size = cfg["input_size"]
    sfreq = cfg["processing"]["frequency"]
    n_channels = cfg["n_channels"]

    # --------------------------------------------------------------
    # Resolve num_workers: if cfg sets -1, use all available cores
    # (or the PyTorch-recommended maximum returned by multiprocessing).
    # --------------------------------------------------------------
    requested_workers = cfg["dataset"].get("num_workers", 0)
    if requested_workers < 0:
        requested_workers = multiprocessing.cpu_count()

    train_ds = TUHDataset(
        train_samples,
        target_len=input_size,
        target_sf=sfreq,
        n_channels=n_channels,
        mean=cfg["dataset"].get("mean"),
        std=cfg["dataset"].get("std"),
    )
    val_ds = TUHDataset(
        val_samples,
        target_len=input_size,
        target_sf=sfreq,
        n_channels=n_channels,
        mean=cfg["dataset"].get("mean"),
        std=cfg["dataset"].get("std"),
    )

    
    persistent = cfg["dataset"]["num_workers"] > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle"],
        num_workers=requested_workers,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=requested_workers,
        persistent_workers=persistent,
    )

    # --------------------------------------------------------------
    # Scheduler will operate per *batch*, so we need the total number
    # of optimisation steps.
    # --------------------------------------------------------------
    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------
    # Prepare checkpoint directory (clear previous run for same model)
    # ------------------------------------------------------------------
    ckpt_root = Path(cfg["checkpoint"]["checkpoint_dir"]).expanduser()
    ckpt_dir = ckpt_root / cfg["model"]["name"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model --------------------------------------------------------------
    model = CwATModel(
        input_size=cfg["input_size"],
        n_channels=n_channels,
        hyp=cfg["model"],
        num_classes=len(cfg["dataset"]["classes"]),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["init_lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    # ------------------------------------------------------------------
    # Scheduler: use cosine annealing (with optional warm-up) so that the
    # learning-rate follows the hyper-params specified in the config.
    # ------------------------------------------------------------------
    scheduler_name = cfg.get("scheduler", {}).get("name", "none").lower()
    if scheduler_name == "cosine":
        eta_min_val = float(cfg["scheduler"].get("lr_min", 0.0))
        total_steps = cfg["train"]["n_epochs"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=eta_min_val,
        )
    else:
        scheduler = None  # fallback to constant LR if not requested

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Logging / TensorBoard ----------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter

    runs_dir = Path(cfg["tensorboard"]["runs_dir"]).expanduser()
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(runs_dir / f"{datetime.now().strftime('%y%m%d%H%M')}_{cfg['model']['name']}")

    # Training loop -----------------------------------------------------------
    epochs = cfg["train"]["n_epochs"]
    global_step = 0
    best_acc = -float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for signals, targets, _ in train_loader:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            # Gradient clipping for training stabilisation
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()/len(train_loader)
            train_acc += accuracy(outputs, targets)/len(train_loader)
            # ------------------------------------------------------------------
            # Per-batch LR scheduler step & logging
            # ------------------------------------------------------------------
            if scheduler is not None:
                scheduler.step()
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # LR is now stepped per batch; nothing to do here

        # Validation ----------------------------------------------------------
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for signals, targets, _ in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                outputs = model(signals)
                val_loss += criterion(outputs, targets).item()
                val_acc += accuracy(outputs, targets)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

        # Save checkpoint -----------------------------------------------------
        if val_acc > best_acc:
            best_acc = val_acc
            #torch.save(model.state_dict(), ckpt_dir / "best.pth")
            logger.info(f"New best model saved with val_acc={best_acc:.4f}")

    writer.close()

    # Return stats for hyper-parameter tuning frameworks (e.g. Optuna)
    return best_acc, ckpt_dir / "best.pth"


# -----------------------------------------------------------------------------
#  Entrypoint -----------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="FILE", help="YAML config file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # Configure logging -------------------------------------------------------
    logs_dir = (Path(__file__).resolve().parent / "../logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{datetime.now().strftime('%y%m%d%H%M')}_{cfg['model']['name']}.log"

    logging.basicConfig(filename=str(log_file), level=logging.INFO, filemode="w")
    logger.addHandler(logging.StreamHandler())

    _best_acc, _best_path = train(cfg) 