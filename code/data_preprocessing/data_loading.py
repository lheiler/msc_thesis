"""Load preprocessed EEG epoch data from pickle files."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def load_data(data_path_base: str, split: str = "train") -> list[tuple[Any, ...]]:
    """Load cleaned epoch data from a pickle file.

    Args:
        data_path_base: Directory containing the pickle files
            (e.g. ``~/thesis/Datasets/tuh-eeg-ab-clean``).
        split: Which split to load (``"train"`` or ``"eval"``).

    Returns:
        List of tuples ``(raw, gender, age, abnormal, sample_id)`` where
        *raw* is an ``mne.io.Raw`` object.

    Raises:
        FileNotFoundError: If the expected pickle file does not exist.
    """
    pickle_file = os.path.join(data_path_base, f"{split}_epochs.pkl")

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")

    logger.info("Loading %s data from %s ...", split, pickle_file)

    with open(pickle_file, "rb") as f:
        epoch_data = pickle.load(f)

    logger.info("Loaded %d samples from %s split", len(epoch_data), split)
    return epoch_data
