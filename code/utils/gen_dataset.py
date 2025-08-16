import torch
import numpy as np
import mne
from pathlib import Path
from torch.utils.data import Dataset
from utils.util import clean_raw_eeg, preprocess_time_domain_input

class TUHFIF60sDataset(torch.utils.data.Dataset):
    """
    Recursively loads .fif EEG files under a root directory (e.g., TUH train/)
    and returns z‑scored segments shaped (C, T) after standard preprocessing.
    Preprocessing: drop A1/A2, pick 19 EEG channels when present, bandpass 3–45 Hz,
    resample to 128 Hz, crop/pad 60 s, z‑score.
    """

    EEG_CHANNELS_19 = [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
        "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
        "Cz", "Pz", "Fz",
    ]

    def __init__(self, root: Path, segment_len_sec: int = 60, target_sfreq: float = 128.0):
        super().__init__()
        self.root = Path(root)
        self.seg_len = segment_len_sec
        self.sfreq = target_sfreq
        self.files = sorted(self.root.rglob("*.fif"))
        if not self.files:
            raise RuntimeError(f"No .fif files found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)
        raw = clean_raw_eeg(raw, segment_length=self.seg_len)
        data = preprocess_time_domain_input(raw, target_sfreq=self.sfreq, segment_len_sec=self.seg_len)
        return torch.from_numpy(data)