"""Descriptive statistics for latent feature datasets."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from torch.utils.data import DataLoader

__all__ = ["compute_dataset_stats"]


def compute_dataset_stats(
    loader: DataLoader,
    *,
    age_bins: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Compute descriptive statistics for a latent-feature DataLoader.

    Args:
        loader: DataLoader whose dataset elements are
            ``(features, gender_code, age_float, abnormal_flag)``.
        age_bins: Bin edges for the age histogram.
            Defaults to ``[0, 10, 20, ..., 80, 120]``.

    Returns:
        Nested dict with sample count, gender/abnormal distributions, and
        age histogram.
    """
    if age_bins is None:
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]

    genders_list: list[float] = []
    ages_list: list[float] = []
    abns_list: list[float] = []

    for _, g, a, ab in loader:
        genders_list.extend(g.detach().cpu().numpy().tolist())
        ages_list.extend(a.detach().cpu().numpy().tolist())
        abns_list.extend(ab.detach().cpu().numpy().tolist())

    genders = np.asarray(genders_list)
    ages = np.asarray(ages_list)
    abns = np.asarray(abns_list)

    gender_values = set(np.unique(genders).tolist())
    if gender_values.issubset({1.0, 2.0}):
        gender_counts = {
            "female(2)": int((genders == 2).sum()),
            "male(1)": int((genders == 1).sum()),
        }
    elif gender_values.issubset({0.0, 1.0}):
        gender_counts = {
            "female(0)": int((genders == 0).sum()),
            "male(1)": int((genders == 1).sum()),
        }
    else:
        gender_counts = {
            "label_0": int((genders == 0).sum()),
            "label_1": int((genders == 1).sum()),
            "label_2": int((genders == 2).sum()),
        }

    stats: dict[str, Any] = {
        "n_samples": int(len(ages)),
        "gender_counts": gender_counts,
        "abnormal_counts": {
            "abnormal(1)": int((abns == 1).sum()),
            "normal(0)": int((abns == 0).sum()),
        },
    }

    bin_labels = [f"{age_bins[i]}-{age_bins[i + 1]}" for i in range(len(age_bins) - 1)]
    stats["age_bin_counts"] = {
        lbl: int(((ages >= age_bins[i]) & (ages < age_bins[i + 1])).sum())
        for i, lbl in enumerate(bin_labels)
    }
    return stats
