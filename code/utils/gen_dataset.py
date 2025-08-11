import torch
import numpy as np
import mne
from pathlib import Path
from torch.utils.data import Dataset

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

        # Drop A1/A2 if present
        for ch in ("A1", "A2"):
            if ch in raw.ch_names:
                raw.drop_channels([ch])

        # Pick intersection with 19-channel set
        picks_19 = [ch for ch in self.EEG_CHANNELS_19 if ch in raw.ch_names]
        if picks_19:
            raw.pick_channels(picks_19)

        # Crop first (works without preload) so we only load 60s
        sf = raw.info["sfreq"]
        #print the length of the recording in seconds
        #print(f"Recording {path} length: {raw.n_times / sf} seconds")
        raw.crop(tmin=0.0, tmax=self.seg_len - 1.0 / sf)

        # Now load data into memory before filtering/resampling
        raw.load_data(verbose=False)

        # Bandpass and resample (requires preload)
        raw.filter(3.0, 45.0, fir_design="firwin", verbose=False)
        if abs(raw.info["sfreq"] - self.sfreq) > 1e-3:
            raw.resample(self.sfreq, npad="auto")

        data = raw.get_data().astype(np.float32)
        tgt_len = int(self.seg_len * self.sfreq)
        if data.shape[1] < tgt_len:
            pad = tgt_len - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
        elif data.shape[1] > tgt_len:
            data = data[:, :tgt_len]

        data = (data - data.mean()) / (data.std() + 1e-8)
        return torch.from_numpy(data)