"""Linearised Jansen-Rit (JR) neural mass model parameter fitting.

Fits a reduced 6-parameter JR model to empirical EEG power spectra via
CMA-ES.  The linearised transfer function is evaluated analytically on
the same Welch frequency grid used for the empirical data.

Public API:
    * ``fit_parameters(freqs, psd, ...)``   - CMA-ES fit on a PSD.
    * ``fit_jr_average_from_raw(raw)``      - Average-PSD convenience wrapper.
    * ``fit_jr_per_channel_from_raw(raw)``  - Per-channel convenience wrapper.
"""
from __future__ import annotations

from typing import Optional, Union

import cma
import mne
import numpy as np

from utils.util import PSD_CALCULATION_PARAMS, compute_psd_from_raw, normalize_psd

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def _jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    numba = type("numba", (), {"jit": _jit})()  # type: ignore[assignment]

mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# Linearised JR transfer functions
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True, fastmath=True)
def _He(jw: complex, A: float, a: float) -> complex:
    """Excitatory synaptic kernel in the frequency domain."""
    return (A * a) / ((jw ** 2) + 2 * a * jw + a ** 2)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _Hi(jw: complex, B: float, b: float) -> complex:
    """Inhibitory synaptic kernel in the frequency domain."""
    return (B * b) / ((jw ** 2) + 2 * b * jw + b ** 2)


@numba.jit(nopython=True, cache=True, fastmath=True)
def _jr_transfer_core(
    jw: complex,
    A: float, a: float,
    B: float, b: float,
    G: float, C1: float,
) -> complex:
    """Core linearised transfer function T(jw) = V(jw) / P(jw)."""
    He = _He(jw, A, a)
    Hi = _Hi(jw, B, b)
    C2 = 0.8 * C1
    C3 = 0.25 * C1
    C4 = 0.25 * C1
    denom = 1.0 - (He ** 2) * (G ** 2) * C1 * C2 + (He * Hi) * (G ** 2) * C3 * C4
    return (He ** 2) * G / denom


def _P_omega(freqs: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Model power |T(jw)|^2 on the given frequency grid.

    Args:
        freqs: Frequency array in Hz.
        p: JR parameter dictionary.

    Returns:
        Power spectrum array aligned with *freqs*.
    """
    w = 2 * np.pi * freqs
    jw_array = 1j * w

    A, a, B, b, G, C1 = p["A"], p["a"], p["B"], p["b"], p["G"], p["C1"]
    T = np.array(
        [_jr_transfer_core(jw, A, a, B, b, G, C1) for jw in jw_array],
        dtype=np.complex128,
    )
    return (np.abs(T) ** 2).astype(float)


# ---------------------------------------------------------------------------
# PSD estimation helper
# ---------------------------------------------------------------------------

def compute_psd(
    raw: mne.io.BaseRaw,
    *,
    channel: Union[str, list[str]] = "O1..",
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs, mean_psd)`` for *raw*.

    Args:
        raw: MNE Raw object.
        channel: Channel name or list to pick.
        l_freq: Low frequency for filtering.
        h_freq: High frequency for filtering.
        fmin: Minimum PSD frequency.
        fmax: Maximum PSD frequency.
        n_fft: FFT length for Welch.

    Returns:
        Tuple of ``(freqs, mean_psd)`` arrays.
    """
    raw_copy = raw.copy().pick([channel] if isinstance(channel, str) else channel)
    spectrum = raw_copy.compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft)
    psds = spectrum.get_data()
    freqs = spectrum.freqs
    return freqs, psds.mean(axis=0)


# ---------------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------------

_PARAM_KEYS: list[str] = [
    "C1",       # base connectivity (C2, C3, C4 are derived)
    "A", "B",   # synaptic gains
    "a", "b",   # synaptic rate constants (s^-1)
    "G",        # linearised sigmoid slope
]


def _dict_to_vector(p: dict[str, float]) -> np.ndarray:
    """Return parameter dict as a NumPy array in canonical order."""
    return np.asarray([p[k] for k in _PARAM_KEYS], dtype=np.float32)


_DEFAULT_THETA0 = np.asarray(
    [135.0, 3.25, 22.0, 100.0, 50.0, 1.5], dtype=float,
)

_DEFAULT_BOUNDS = np.asarray(
    [
        (50.0, 300.0),   # C1
        (1.0, 10.0),     # A
        (5.0, 60.0),     # B
        (50.0, 150.0),   # a
        (20.0, 120.0),   # b
        (0.1, 5.0),      # G
    ],
    dtype=float,
)


def _loss_function(
    theta: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
) -> float:
    """MSE between model and empirical PSD on identical Welch bins."""
    p_full = dict(zip(_PARAM_KEYS, theta))
    model_psd = _P_omega(freqs, p_full)

    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", 45.0))
    mask = (freqs >= fmin) & (freqs <= fmax)

    return float(
        np.mean((normalize_psd(model_psd[mask]) - normalize_psd(real_psd[mask])) ** 2)
    )


class _LossFunction:
    """Pickleable CMA-ES objective wrapping ``_loss_function``."""

    def __init__(self, freqs: np.ndarray, psd: np.ndarray) -> None:
        self.freqs = freqs
        self.psd = psd

    def __call__(self, x: np.ndarray) -> float:
        return _loss_function(np.asarray(x, dtype=float), self.freqs, self.psd)


def fit_parameters(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    initial_theta: Optional[np.ndarray] = None,
    sigma0: float = 0.5,
    bounds: Optional[np.ndarray] = None,
    cma_opts: Optional[dict] = None,
    return_full: bool = False,
) -> Union[dict[str, float], tuple[dict[str, float], np.ndarray, float]]:
    """Fit JR parameters to a power spectrum using CMA-ES.

    Args:
        freqs: 1-D frequency array (Hz).
        psd: 1-D power array matching *freqs*.
        initial_theta: Optional 6-element starting vector.
        sigma0: Initial CMA-ES sampling spread.
        bounds: ``(6, 2)`` lower/upper bounds array.
        cma_opts: Extra options forwarded to ``cma.CMAEvolutionStrategy``.
        return_full: If True, also return ``(theta_best, loss_best)``.

    Returns:
        Best-fit parameter dict (and optionally raw vector + loss).
    """
    theta0 = _DEFAULT_THETA0 if initial_theta is None else np.asarray(initial_theta, dtype=float)
    bounds_arr = _DEFAULT_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (len(_PARAM_KEYS), 2):
        raise ValueError(f"bounds must have shape ({len(_PARAM_KEYS)}, 2)")

    lower_bounds, upper_bounds = bounds_arr[:, 0], bounds_arr[:, 1]
    opts: dict = {
        "bounds": [lower_bounds.tolist(), upper_bounds.tolist()],
        "verbose": -9,
        "verb_log": 0,
        "tolfun": 1e-4,
        "maxiter": 600,
        "seed": 42,
    }
    if cma_opts:
        opts.update(cma_opts)

    es = cma.CMAEvolutionStrategy(theta0.tolist(), sigma0, opts)
    es.optimize(_LossFunction(freqs, psd), n_jobs=1)

    theta_best = np.asarray(es.result.xbest, dtype=float)
    best_params = dict(zip(_PARAM_KEYS, theta_best))

    if return_full:
        return best_params, theta_best, float(es.result.fbest)
    return best_params


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def fit_jr_average_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit JR to the channel-averaged PSD.

    Args:
        raw: MNE Raw object.
        **fit_kwargs: Forwarded to ``fit_parameters``.

    Returns:
        1-D float32 parameter vector in ``_PARAM_KEYS`` order.
    """
    psd, freqs = compute_psd_from_raw(
        raw, calculate_average=True, normalize=False, return_freqs=True,
    )
    return _dict_to_vector(fit_parameters(freqs, psd, **fit_kwargs))


def fit_jr_per_channel_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit JR per channel and return a concatenated parameter vector.

    Args:
        raw: MNE Raw object.
        **fit_kwargs: Forwarded to ``fit_parameters``.

    Returns:
        1-D float32 array of all per-channel parameters concatenated.
    """
    psd_matrix, freqs = compute_psd_from_raw(
        raw, calculate_average=False, normalize=False, return_freqs=True,
    )
    all_params = [
        _dict_to_vector(fit_parameters(freqs, row, **fit_kwargs))
        for row in psd_matrix
    ]
    return np.concatenate(all_params, axis=0)
