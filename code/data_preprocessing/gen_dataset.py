"""PyTorch dataset for loading preprocessed TUH EEG pickle files."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Union

import torch
from torch.utils.data import Dataset

from utils.util import preprocess_time_domain_input

logger = logging.getLogger(__name__)

__all__ = ["TUHFIF60sDataset"]

DEFAULT_SFREQ = 128.0
DEFAULT_SEGMENT_LEN_SEC = 10.0


class TUHFIF60sDataset(Dataset):
    """Dataset that loads a single ``*_epochs.pkl`` file and yields ``(C, T)`` tensors.

    Each record in the pickle is expected to be
    ``(raw, gender, age, abnormal_flag, sample_id)`` as produced by the
    cleaning pipeline.

    Args:
        pkl_path: Path to the pickle file containing EEG records.
    """

    def __init__(self, pkl_path: Union[os.PathLike, str]) -> None:
        super().__init__()
        self.pkl_path = str(pkl_path)
        with open(self.pkl_path, "rb") as f:
            records = pickle.load(f)

        self._records: list[tuple] = records

        try:
            self.sample_ids = [str(r[4]) for r in records]
            self.genders = [int(r[1]) for r in records]
            self.ages = [int(r[2]) for r in records]
            self.labels = [int(r[3]) for r in records]
        except (IndexError, TypeError, ValueError):
            logger.warning("Could not parse metadata from records in %s", pkl_path)
            self.sample_ids = None
            self.genders = None
            self.ages = None
            self.labels = None

        self.sfreq: float = DEFAULT_SFREQ
        self.seg_len: float = DEFAULT_SEGMENT_LEN_SEC

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw = self._records[idx][0]
        x = preprocess_time_domain_input(
            raw, target_sfreq=self.sfreq, segment_len_sec=self.seg_len,
        )
        return torch.from_numpy(x)
