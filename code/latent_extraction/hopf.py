"""Hopf (Stuart-Landau) spectral proxy for EEG latent features.

For each canonical frequency band, fits a Lorentzian peak on a flat
baseline via grid-search + least squares:

    S(f) ~ A / ((f - f0)^2 + gamma^2) + b

Returns ``[A, f0, gamma, b]`` per band, concatenated across bands
(and optionally across channels).
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Sequence, Tuple, Union

import mne
import numpy as np

from utils.util import compute_psd_from_raw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
Band = Tuple[float, float]
DEFAULT_BANDS: tuple[tuple[str, Band], ...] = (
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
)

PARAMS_PER_BAND = 4
MIN_WELCH_BINS = 8


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def _fit_lorentzian_band(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_lo: float,
    f_hi: float,
    *,
    n_f0: int = 60,
    n_gamma: int = 40,
    gamma_min: float = 0.2,
    gamma_max: float = 8.0,
) -> tuple[float, float, float, float]:
    """Fit Lorentzian parameters (A, f0, gamma, b) on a frequency window.

    Args:
        freqs: Full frequency array.
        psd: PSD values corresponding to *freqs*.
        f_lo: Lower band edge (Hz).
        f_hi: Upper band edge (Hz).
        n_f0: Grid points for centre frequency search.
        n_gamma: Grid points for width search.
        gamma_min: Minimum Lorentzian width.
        gamma_max: Maximum Lorentzian width.

    Returns:
        Tuple ``(A, f0, gamma, b)`` of best-fit parameters.

    Raises:
        RuntimeError: If too few Welch bins are available in the band.
    """
    i0 = int(np.searchsorted(freqs, f_lo, side="left"))
    i1 = int(np.searchsorted(freqs, f_hi, side="right"))
    if (i1 - i0) < MIN_WELCH_BINS:
        deficit = MIN_WELCH_BINS - (i1 - i0)
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        i0 = max(0, i0 - pad_left)
        i1 = min(len(freqs), i1 + pad_right)
        if (i1 - i0) < MIN_WELCH_BINS and len(freqs) >= MIN_WELCH_BINS:
            i0 = 0
            i1 = min(len(freqs), MIN_WELCH_BINS)

    f = freqs[i0:i1].astype(np.float64)
    y = psd[i0:i1].astype(np.float64)
    if f.size < 3:
        raise RuntimeError(
            f"Insufficient Welch bins to fit Lorentzian around [{f_lo},{f_hi}]."
        )

    f0_grid = np.linspace(max(f_lo + 0.1, f.min()), min(f_hi - 0.1, f.max()), n_f0)
    gamma_grid = np.linspace(gamma_min, gamma_max, n_gamma)

    one = np.ones_like(f)
    best_sse = np.inf
    best_params = (0.0, float(f0_grid[len(f0_grid) // 2]), float(gamma_grid[len(gamma_grid) // 2]), 0.0)

    for f0 in f0_grid:
        df2 = (f - f0) ** 2
        for gamma in gamma_grid:
            k = 1.0 / (df2 + gamma * gamma)
            X = np.column_stack([k, one])
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            A, b = float(theta[0]), float(theta[1])
            if not np.isfinite(A) or not np.isfinite(b) or A < 0.0 or b < 0.0:
                continue
            resid = y - (A * k + b)
            sse = float(resid @ resid)
            if sse < best_sse:
                best_sse = sse
                best_params = (A, float(f0), float(gamma), b)

    return best_params


def _fit_channel_bands(
    args: tuple[int, np.ndarray, np.ndarray, Sequence[tuple[str, Band]]],
) -> tuple[int, list[float]]:
    """Fit all bands for one channel (used by ProcessPoolExecutor)."""
    channel_idx, psd_channel, freqs, bands = args
    params: list[float] = []
    for _name, (flo, fhi) in bands:
        A, f0, gamma, b = _fit_lorentzian_band(freqs, psd_channel, flo, fhi)
        params.extend([A, f0, gamma, b])
    return channel_idx, params


def fit_hopf_from_raw(
    raw: mne.io.BaseRaw,
    *,
    bands: Sequence[tuple[str, Band]] = DEFAULT_BANDS,
    psd_kwargs: Optional[dict] = None,
    per_channel: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """Return Hopf spectral parameters from an MNE Raw.

    Args:
        raw: Continuous EEG recording (pre-cleaned).
        bands: Frequency bands to fit (name, (f_low, f_high)).
        psd_kwargs: (Unused, kept for API compatibility.)
        per_channel: If True, fit each channel separately.
        n_jobs: Number of parallel workers (-1 = all cores).

    Returns:
        Float32 feature vector ``[A, f0, gamma, b]`` per band
        (and per channel if *per_channel* is True).
    """
    if per_channel:
        avg_psd, freqs = compute_psd_from_raw(
            raw, calculate_average=False, normalize=True, return_freqs=True,
        )
    else:
        avg_psd, freqs = compute_psd_from_raw(
            raw, calculate_average=True, normalize=True, return_freqs=True,
        )
    avg_psd = avg_psd.astype(np.float64)

    params: list[float] = []

    if per_channel and avg_psd.ndim == 2 and len(avg_psd) > 1:
        n_cores = max(1, n_jobs) if n_jobs != -1 else None
        args_list = [
            (i, avg_psd[i], freqs, bands) for i in range(len(avg_psd))
        ]

        try:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = list(executor.map(_fit_channel_bands, args_list))
            results.sort(key=lambda x: x[0])
            for _idx, channel_params in results:
                params.extend(channel_params)
        except (OSError, RuntimeError) as exc:
            logger.warning("Parallel processing failed (%s), falling back to sequential.", exc)
            for i in range(len(avg_psd)):
                for _name, (flo, fhi) in bands:
                    A, f0, gamma, b = _fit_lorentzian_band(freqs, avg_psd[i], flo, fhi)
                    params.extend([A, f0, gamma, b])
    elif per_channel:
        psd_1d = avg_psd[0] if avg_psd.ndim == 2 else avg_psd
        for _name, (flo, fhi) in bands:
            A, f0, gamma, b = _fit_lorentzian_band(freqs, psd_1d, flo, fhi)
            params.extend([A, f0, gamma, b])
    else:
        for _name, (flo, fhi) in bands:
            A, f0, gamma, b = _fit_lorentzian_band(freqs, avg_psd, flo, fhi)
            params.extend([A, f0, gamma, b])

    return np.asarray(params, dtype=np.float32).flatten()


def hopf_feature_names(
    bands: Sequence[tuple[str, Band]] = DEFAULT_BANDS,
) -> list[str]:
    """Return human-readable names for each element of the feature vector."""
    names: list[str] = []
    for name, _ in bands:
        names.extend([f"{name}_A", f"{name}_f0", f"{name}_gamma", f"{name}_b"])
    return names


def hopf_feature_dim(
    bands: Sequence[tuple[str, Band]] = DEFAULT_BANDS,
) -> int:
    """Return the length of the feature vector."""
    return PARAMS_PER_BAND * len(tuple(bands))


__all__ = [
    "fit_hopf_from_raw",
    "hopf_feature_dim",
    "hopf_feature_names",
]
