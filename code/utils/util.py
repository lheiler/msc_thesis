"""Shared utilities for PSD computation, normalization, and preprocessing.

Provides canonical Welch PSD parameters, channel montage definitions,
device selection, and JSONL serialization helpers used across the pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

logger = logging.getLogger(__name__)

STANDARD_EEG_CHANNELS: list[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]

PSD_CALCULATION_PARAMS: dict[str, float] = {
    "n_fft": 512,
    "n_overlap": 256,
    "n_per_seg": 512,
    "min_freq": 1,
    "max_freq": 45.0,
    "segment_length": 10.0,
    "sfreq": 128.0,
}


def select_device() -> torch.device:
    """Select the best available compute device (CUDA > MPS > CPU).

    Returns:
        ``torch.device`` for the selected backend.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def ensure_float32_tensor(x: Any) -> torch.Tensor:
    """Convert an array-like to a ``torch.float32`` tensor.

    Args:
        x: Input array, list, or tensor.

    Returns:
        Float32 tensor.
    """
    return torch.as_tensor(x, dtype=torch.float32)


def make_latent_record(
    latent_feature: Any,
    gender: Any,
    age: Any,
    abnormal: Any,
    sample_id: Any,
) -> tuple[list[float], int, int, int, str]:
    """Serialize a latent-feature sample into JSONL-compatible tuple.

    Args:
        latent_feature: Feature vector (array-like or tensor).
        gender: Gender label scalar.
        age: Age label scalar.
        abnormal: Abnormality label scalar.
        sample_id: Unique sample identifier.

    Returns:
        5-tuple ``(vec, gender, age, abnormal, sample_id)``.
    """
    vec = latent_feature.tolist() if hasattr(latent_feature, "tolist") else latent_feature
    g = int(gender.item()) if hasattr(gender, "item") else int(gender)
    a = int(age.item()) if hasattr(age, "item") else int(age)
    ab = int(abnormal.item()) if hasattr(abnormal, "item") else int(abnormal)
    return (vec, g, a, ab, str(sample_id))


def truncate_file(path: str) -> None:
    """Create or truncate a file to zero bytes.

    Args:
        path: File path to create/truncate.
    """
    with open(path, "w"):
        pass


def append_jsonl(path: str, record_tuple: tuple) -> None:
    """Append a JSON-serialized record to a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.
        record_tuple: Tuple to serialize and append.
    """
    with open(path, "a") as f:
        f.write(json.dumps(record_tuple) + "\n")


def normalize_psd(psd: np.ndarray) -> np.ndarray:
    """Log-transform and z-score a PSD array.

    Handles both single-channel (1-D) and multi-channel (2-D) inputs.
    Replaces NaN/Inf values with safe defaults before normalization.

    Args:
        psd: Raw PSD values, shape ``(F,)`` or ``(C, F)``.

    Returns:
        Normalized PSD with the same shape as input.
    """
    if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
        logger.warning("PSD contains NaN/Inf values, replacing with safe defaults")
        psd = np.nan_to_num(psd, nan=1.0, posinf=1e6, neginf=1e-12)

    epsilon = 1e-8 * np.max(psd) if np.max(psd) > 0 else 1e-8
    log_psd = np.log10(psd + epsilon)

    if psd.ndim == 1:
        std_val = log_psd.std()
        if std_val < 1e-8:
            return log_psd - log_psd.mean()
        return (log_psd - log_psd.mean()) / std_val
    else:
        means = log_psd.mean(axis=1, keepdims=True)
        stds = log_psd.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1.0
        return (log_psd - means) / stds


def normalize_psd_torch(psd: torch.Tensor) -> torch.Tensor:
    """Log-transform and z-score a PSD tensor (PyTorch variant).

    Args:
        psd: Raw PSD values, shape ``(F,)`` or ``(C, F)``.

    Returns:
        Normalized PSD tensor with the same shape as input.
    """
    if torch.any(torch.isnan(psd)) or torch.any(torch.isinf(psd)):
        logger.warning("PSD contains NaN/Inf values, replacing with safe defaults")
        psd = torch.nan_to_num(psd, nan=1.0, posinf=1e6, neginf=1e-12)

    epsilon = 1e-8 * torch.max(psd) if torch.max(psd) > 0 else 1e-8
    log_psd = torch.log10(psd + epsilon)

    if psd.ndim == 1:
        std_val = log_psd.std()
        if std_val < 1e-8:
            return log_psd - log_psd.mean()
        return (log_psd - log_psd.mean()) / std_val
    else:
        means = log_psd.mean(dim=1, keepdim=True)
        stds = log_psd.std(dim=1, keepdim=True)
        stds = torch.clamp(stds, min=1e-8)
        return (log_psd - means) / stds


def compute_psd_from_raw(
    raw: mne.io.BaseRaw,
    *,
    n_fft: int = PSD_CALCULATION_PARAMS["n_fft"],
    n_overlap: int = PSD_CALCULATION_PARAMS["n_overlap"],
    n_per_seg: int = PSD_CALCULATION_PARAMS["n_per_seg"],
    calculate_average: bool = False,
    normalize: bool = True,
    return_freqs: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute Welch PSD from an MNE Raw object.

    Args:
        raw: MNE Raw object (assumed already channel-cleaned and bandpassed).
        n_fft: FFT length.
        n_overlap: Overlap between segments.
        n_per_seg: Samples per Welch segment.
        calculate_average: If True, average across channels -> ``(F,)``.
        normalize: If True, apply ``normalize_psd`` (log10 + z-score).
        return_freqs: If True, also return frequency bins.

    Returns:
        PSD array ``(F,)`` or ``(C, F)``, optionally with frequency array.
    """
    sfreq = float(raw.info.get("sfreq", 128.0))
    data = raw.get_data()
    psd_data, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        average="mean",
        verbose=False,
        fmin=PSD_CALCULATION_PARAMS["min_freq"],
        fmax=PSD_CALCULATION_PARAMS["max_freq"],
    )

    if calculate_average:
        psd_out = psd_data.mean(axis=0).astype(np.float32)
    else:
        psd_out = psd_data.astype(np.float32)

    if normalize:
        psd_out = normalize_psd(psd_out)

    if return_freqs:
        return psd_out, freqs.astype(np.float32)
    return psd_out


def compute_psd_from_array(
    y: np.ndarray,
    *,
    sfreq: float,
    n_fft: int = PSD_CALCULATION_PARAMS["n_fft"],
    n_overlap: int = PSD_CALCULATION_PARAMS["n_overlap"],
    n_per_seg: int = PSD_CALCULATION_PARAMS["n_per_seg"],
    normalize: bool = True,
    return_freqs: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute Welch PSD for a 1-D time series.

    Uses the same Welch parameters as ``compute_psd_from_raw`` to ensure
    identical frequency grids.

    Args:
        y: 1-D array of time-domain samples.
        sfreq: Sampling frequency in Hz.
        n_fft: FFT length.
        n_overlap: Overlap between segments.
        n_per_seg: Samples per Welch segment.
        normalize: If True, apply ``normalize_psd``.
        return_freqs: If True, also return frequency bins.

    Returns:
        PSD array ``(F,)`` float32, optionally with frequency array.
    """
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim != 1:
        y_np = y_np.reshape(-1)
    psd_arr, freqs = mne.time_frequency.psd_array_welch(
        y_np[None, :],
        sfreq=float(sfreq),
        n_fft=int(n_fft),
        n_overlap=int(n_overlap),
        n_per_seg=int(n_per_seg),
        average="mean",
        verbose=False,
        fmin=PSD_CALCULATION_PARAMS["min_freq"],
        fmax=PSD_CALCULATION_PARAMS["max_freq"],
    )
    psd_vec = psd_arr[0].astype(np.float32)
    if normalize:
        psd_vec = normalize_psd(psd_vec)
    if return_freqs:
        return psd_vec, freqs.astype(np.float32)
    return psd_vec


def preprocess_time_domain_input(
    raw: mne.io.BaseRaw,
    *,
    target_sfreq: float = 128.0,
    segment_len_sec: int = 10,
) -> np.ndarray:
    """Resample, crop/pad, and z-score raw data for time-domain models.

    Assumes channels are already cleaned, ordered, and bandpass-filtered
    upstream.

    Args:
        raw: MNE Raw object.
        target_sfreq: Target sampling frequency in Hz.
        segment_len_sec: Desired segment length in seconds.

    Returns:
        Z-scored array of shape ``(C, T)`` where ``T = segment_len_sec * target_sfreq``.
    """
    x = raw.copy()
    x.load_data(verbose=False)
    sfreq_curr = float(x.info.get("sfreq", target_sfreq))
    tmax = min(segment_len_sec, x.times[-1])
    x.crop(tmin=0.0, tmax=tmax - 1.0 / sfreq_curr)
    if abs(sfreq_curr - target_sfreq) > 1e-3:
        x.resample(target_sfreq, npad="auto")
    data = x.get_data().astype(np.float32)
    tgt_len = int(segment_len_sec * target_sfreq)
    if data.shape[1] < tgt_len:
        pad = tgt_len - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
    elif data.shape[1] > tgt_len:
        data = data[:, :tgt_len]
        logger.debug("Truncated %d samples to %d", data.shape[1], tgt_len)
    data = (data - data.mean()) / (data.std() + 1e-8)
    return data


def _to_numpy_1d(x: Any) -> np.ndarray:
    """Convert a tensor or array to a 1-D NumPy array.

    Args:
        x: Input tensor, array, or array-like. If 2-D, takes the first row.

    Returns:
        1-D NumPy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    return x


def plot_psd(psd: Any, freqs: Any, path: str) -> None:
    """Plot a single PSD curve and save to disk.

    Args:
        psd: PSD values (1-D array-like).
        freqs: Frequency bins (1-D array-like).
        path: Output file path for the saved figure.
    """
    x = _to_numpy_1d(psd)
    f = _to_numpy_1d(freqs)
    plt.figure()
    plt.plot(f, x)
    plt.xlabel("Hz")
    plt.ylabel("PSD (norm)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_psd_comparison(
    psd1: Any, psd2: Any, freqs: Any, path: str
) -> None:
    """Plot two PSD curves overlaid and save to disk.

    Args:
        psd1: First PSD (1-D array-like).
        psd2: Second PSD (1-D array-like).
        freqs: Frequency bins (1-D array-like).
        path: Output file path for the saved figure.
    """
    x1 = _to_numpy_1d(psd1)
    x2 = _to_numpy_1d(psd2)
    f = _to_numpy_1d(freqs)
    plt.figure()
    plt.plot(f, x1, label="PSD 1")
    plt.plot(f, x2, label="PSD 2")
    plt.xlabel("Hz")
    plt.ylabel("PSD (norm)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
