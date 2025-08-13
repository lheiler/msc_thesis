

"""
Hopf-based latent feature extraction from EEG.

This implements a simple stochastic Hopf (Stuart–Landau) spectral proxy.
For each canonical band, it fits a Lorentzian peak sitting on a flat baseline:
    S(f) ≈ A / ((f - f0)^2 + gamma^2) + b
on the *channel-averaged* PSD from MNE's Welch estimate.

Returned feature vector is [A, f0, gamma, b] per band, concatenated in a fixed order.
This yields 4×(#bands) parameters, typically 16 for delta–beta.

Usage in your extractor:
    from latent_extraction.hopf import fit_hopf_from_raw
    ...
    elif method == "hopf":
        latent_feature = fit_hopf_from_raw(x, as_vector=True)

No external dependencies beyond numpy and mne.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np

try:
    import mne  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "latant_extraction.hopf requires MNE available at runtime. "
        "Install mne or ensure it is on PYTHONPATH."
    ) from e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
Band = Tuple[float, float]
DEFAULT_BANDS: Tuple[Tuple[str, Band], ...] = (
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta",  (13.0, 30.0)),
)

DEFAULT_PSD_KW = dict(
    method="welch",
    fmin=1.0,
    fmax=30.0,
    n_per_seg=512,
    n_fft=512,
    verbose="ERROR",
)

# -----------------------------------------------------------------------------
# Core fitting
# -----------------------------------------------------------------------------

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
) -> Tuple[float, float, float, float]:
    """Fit A, f0, gamma, b on a frequency window by grid-search + least squares.

    Model: y ≈ A * k + b, where k = 1 / ((f - f0)^2 + gamma^2).
    For each (f0, gamma) on grids, solve linear least squares for (A, b),
    reject negative A or b, and keep the minimum SSE.
    """
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    f = freqs[mask].astype(np.float64)
    y = psd[mask].astype(np.float64)

    if f.size < 8:
        raise RuntimeError(
            f"Not enough frequency bins in [{f_lo},{f_hi}] to fit Lorentzian. "
            f"Increase n_fft/n_per_seg or widen the band."
        )

    f0_grid = np.linspace(max(f_lo + 0.1, f.min()), min(f_hi - 0.1, f.max()), n_f0)
    gamma_grid = np.linspace(gamma_min, gamma_max, n_gamma)

    one = np.ones_like(f)
    best_sse = np.inf
    best_params = (0.0, float(f0_grid[len(f0_grid)//2]), float(gamma_grid[len(gamma_grid)//2]), 0.0)

    for f0 in f0_grid:
        # Precompute (f - f0)^2 term once per f0
        df2 = (f - f0) ** 2
        for gamma in gamma_grid:
            k = 1.0 / (df2 + gamma * gamma)
            # Linear least squares for A and b
            X = np.column_stack([k, one])
            theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            A, b = float(theta[0]), float(theta[1])
            # Enforce non-negativity and finite values
            if not np.isfinite(A) or not np.isfinite(b) or A < 0.0 or b < 0.0:
                continue
            resid = y - (A * k + b)
            sse = float(resid @ resid)
            if sse < best_sse:
                best_sse = sse
                best_params = (A, float(f0), float(gamma), b)

    A, f0, gamma, b = best_params
    return A, f0, gamma, b


def fit_hopf_from_raw(
    raw: "mne.io.BaseRaw",
    *,
    bands: Sequence[Tuple[str, Band]] = DEFAULT_BANDS,
    psd_kwargs: Dict[str, Union[float, int, str]] | None = None,
    as_vector: bool = True,
) -> Union[np.ndarray, Dict[str, Dict[str, float]]]:
    """Return Hopf spectral parameters from an MNE Raw.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording. Channels should be pre-cleaned and selected.
    bands : sequence of (name, (f_low, f_high))
        Frequency bands to fit. Defaults to delta, theta, alpha, beta.
    psd_kwargs : dict
        Passed to `raw.compute_psd`. Defaults set in DEFAULT_PSD_KW.
    as_vector : bool
        If True return a float32 vector [A,f0,gamma,b]_per_band.
        If False return a dict per band.
    """
    kw = DEFAULT_PSD_KW.copy()
    if psd_kwargs:
        kw.update(psd_kwargs)

    psd = raw.compute_psd(**kw)
    psds, freqs = psd.get_data(return_freqs=True)
    # Average across channels to obtain a global spectrum to fit
    avg_psd = psds.mean(axis=0).astype(np.float64)

    out_dict: Dict[str, Dict[str, float]] = {}
    params: List[float] = []

    for name, (flo, fhi) in bands:
        A, f0, gamma, b = _fit_lorentzian_band(freqs, avg_psd, flo, fhi)
        out_dict[name] = {"A": float(A), "f0": float(f0), "gamma": float(gamma), "b": float(b)}
        params.extend([A, f0, gamma, b])

    if as_vector:
        return np.asarray(params, dtype=np.float32)
    return out_dict


def hopf_feature_names(
    bands: Sequence[Tuple[str, Band]] = DEFAULT_BANDS,
) -> List[str]:
    """Names for each element of the returned vector from `fit_hopf_from_raw`."""
    names: List[str] = []
    for name, _ in bands:
        names.extend([f"{name}_A", f"{name}_f0", f"{name}_gamma", f"{name}_b"])
    return names


def hopf_feature_dim(
    bands: Sequence[Tuple[str, Band]] = DEFAULT_BANDS,
) -> int:
    """Length of the returned feature vector."""
    return 4 * len(tuple(bands))


__all__ = [
    "fit_hopf_from_raw",
    "hopf_feature_dim",
    "hopf_feature_names",
]
