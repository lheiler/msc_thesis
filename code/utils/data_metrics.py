from typing import Dict, Any
import numpy as np

__all__ = ["compute_dataset_stats"]

def compute_dataset_stats(loader, *, age_bins=None) -> Dict[str, Any]:
    """Return descriptive statistics for a DataLoader.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        The dataloader whose dataset elements are
        (features, gender_code, age_float, abnormal_flag).
    age_bins : list[int] | None
        Optional list of bin edges for age histogram. Default:
        [0,10,20,30,40,50,60,70,80,120].
    Returns
    -------
    dict
        Nested dict with counts per gender, abnormal ratio, and
        histogram of ages.
    """
    if age_bins is None:
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]

    genders = []
    ages = []
    abns = []
    for _, g, a, ab in loader:
        # assuming tensors
        genders.extend(g.detach().cpu().numpy().tolist())
        ages.extend(a.detach().cpu().numpy().tolist())
        abns.extend(ab.detach().cpu().numpy().tolist())

    genders = np.asarray(genders)
    ages = np.asarray(ages)
    abns = np.asarray(abns)

    stats = {
        "n_samples": int(len(ages)),
        "gender_counts": {
            "female(2)": int((genders == 2).sum()),
            "male(1)": int((genders == 1).sum()),
            "unknown(0)": int((genders == 0).sum()),
        },
        "abnormal_counts": {
            "abnormal(1)": int((abns == 1).sum()),
            "normal(0)": int((abns == 0).sum()),
        },
    }

    bin_labels = [f"{age_bins[i]}–{age_bins[i+1]}" for i in range(len(age_bins) - 1)]
    age_bin_counts = {
        lbl: int(((ages >= age_bins[i]) & (ages < age_bins[i + 1])).sum())
        for i, lbl in enumerate(bin_labels)
    }
    stats["age_bin_counts"] = age_bin_counts
    return stats 