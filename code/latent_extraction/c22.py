"""Catch22 feature extraction for EEG signals.

Extracts the 22 canonical time-series features per channel using
``pycatch22``, concatenated across all standard EEG channels.
"""
from __future__ import annotations

import logging

import mne
import numpy as np
import pycatch22

from utils.util import STANDARD_EEG_CHANNELS

logger = logging.getLogger(__name__)

CATCH22_FEATURE_DIM = 22


def extract_values(features: dict) -> np.ndarray:
    """Extract the value array from a Catch22 feature dict.

    Args:
        features: Output of ``pycatch22.catch22_all()``.

    Returns:
        1-D float32 array of 22 feature values.
    """
    return np.array(features["values"], dtype=np.float32)


def extract_c22(x: mne.io.BaseRaw) -> np.ndarray:
    """Extract Catch22 features from all standard EEG channels.

    Args:
        x: MNE Raw object with EEG data.

    Returns:
        1-D float32 array of shape ``(19 * 22,)`` (channels x features).
    """
    all_features: list[np.ndarray] = []
    for ch in STANDARD_EEG_CHANNELS:
        if ch in x.ch_names:
            ch_data = x.copy().pick([ch]).get_data()[0]
            features = pycatch22.catch22_all(ch_data)
            all_features.append(extract_values(features))
        else:
            logger.warning("Missing channel: %s - padding with zeros", ch)
            all_features.append(np.zeros(CATCH22_FEATURE_DIM, dtype=np.float32))
    return np.concatenate(all_features, axis=0)
