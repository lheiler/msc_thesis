import tqdm
from pathlib import Path
import sys, os, time, logging, argparse, yaml, math, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from datetime import datetime
from typing import List, Tuple
from collections import OrderedDict

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
from models.encoder import res_encoderM  # noqa: E402
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
        raw, _sex_code, abn = self.samples[idx]
        signal = self._process_raw(raw)

        if self.mean is not None and self.std is not None:
            signal = z_normalise(signal, self.mean, self.std)

        # Build a stable file identifier for per-case evaluation (filename stem)
        fname = os.path.basename(raw.filenames[0]) if raw.filenames else f"sample_{idx}.fif"
        return signal, torch.tensor(abn, dtype=torch.long), fname

# -----------------------------------------------------------------------------
#  Model wrapper (identical to original but renamed for clarity)
# -----------------------------------------------------------------------------

class CwATModel(nn.Module):
    def __init__(self, input_size: int, n_channels: int, hyp: dict, num_classes: int):
        super().__init__()
        self.encoder = res_encoderM(
            n_channels=n_channels,
            groups=n_channels,
            num_classes=num_classes,
            d_model=hyp["d_model"],
        )
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

    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=cfg["dataset"]["shuffle"],
        num_workers=cfg["dataset"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
    )

    # Build model --------------------------------------------------------------
    model = CwATModel(
        input_size=cfg["input_size"],
        n_channels=n_channels,
        hyp=cfg["model"],
        num_classes=len(cfg["dataset"]["classes"]),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["init_lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Logging / TensorBoard ----------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter

    runs_dir = Path(cfg["tensorboard"]["runs_dir"]).expanduser()
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(runs_dir / f"{datetime.now().strftime('%y%m%d%H%M')}_{cfg['model']['name']}")

    # Training loop -----------------------------------------------------------
    epochs = cfg["train"]["n_epochs"]
    global_step = 0
    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        for signals, targets, _ in train_loader:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Adjust LR -----------------------------------------------------------
        lr = poly_lr_scheduler(optimizer, cfg["optimizer"]["init_lr"], epoch + 1, max_iter=epochs)
        writer.add_scalar("train/lr", lr, epoch)

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
        logger.info(f"Epoch {epoch+1}/{epochs} – val_loss: {val_loss:.4f} – val_acc: {val_acc:.4f}")

        # Save checkpoint -----------------------------------------------------
        ckpt_dir = Path(cfg["checkpoint"]["checkpoint_dir"]).expanduser()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"{datetime.now().strftime('%y%m%d%H%M')}_{cfg['model']['name']}.pth")

    writer.close()


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

    train(cfg) 