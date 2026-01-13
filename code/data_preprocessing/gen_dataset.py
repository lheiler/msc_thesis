import torch
import numpy as np
import mne
import os
import pickle
from pathlib import Path
from torch.utils.data import Dataset
import sys

utils_path = Path(__file__).resolve().parent.parent / "utils"
sys.path.insert(0, str(utils_path))

from util import PSD_CALCULATION_PARAMS, preprocess_time_domain_input


class TUHFIF60sDataset(torch.utils.data.Dataset):
    """
    Minimal dataset that loads a single *_epochs.pkl file and yields (C, T)
    tensors from contained MNE Raw objects. No validations, no preprocessing.

    Each record in the pickle is expected to be (raw, g, a, ab, sample_id),
    as produced by the cleaning pipeline.
    """

    def __init__(self, pkl_path: os.PathLike | str) -> None:
        super().__init__()
        self.pkl_path = str(pkl_path)
        with open(self.pkl_path, "rb") as f:
            records = pickle.load(f)

        self._records = records  # list of 5-tuples
        # Optional convenience attributes
        try:
            self.sample_ids = [str(r[4]) for r in records]
            self.genders = [int(r[1]) for r in records]
            self.ages = [int(r[2]) for r in records]
            self.labels = [int(r[3]) for r in records]
        except Exception:
            self.sample_ids, self.genders, self.ages, self.labels = None, None, None, None

        # Best-effort exposure of sfreq/seg_len
        self.sfreq = 128.0
        self.seg_len = 10.0

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw = self._records[idx][0]
        x = preprocess_time_domain_input(raw, target_sfreq=self.sfreq, segment_len_sec=self.seg_len)
        return torch.from_numpy(x)