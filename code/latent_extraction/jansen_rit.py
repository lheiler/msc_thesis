"""jansen_rit.py

Linearized Jansen–Rit (JR) neural mass model parameter-fitting utilities.

Public API mirrors the CTM helper for consistency:

    * compute_psd(raw, fmin=1.0, fmax=40.0, n_fft=128)
        – Returns (freqs, mean_psd) from an MNE Raw instance.

    * fit_parameters(freqs, psd, *, initial_theta=None, sigma0=0.5,
                     bounds=None, cma_opts=None, return_full=False)
        – Runs CMA-ES to fit JR parameters (reduced 6‑parameter form) to the
          supplied power spectrum. Returns a dict of best‑fit parameters. If
          ``return_full=True`` also returns (theta_best, loss_best).

    * fit_jr_from_raw(raw, **kwargs)
        – Convenience wrapper: estimate PSD from all EEG channels, average,
          and fit the JR model. All kwargs pass to :func:`fit_parameters`.
"""

from __future__ import annotations

import numpy as np
import cma
import mne
import os
import re
from utils.util import (
    normalize_psd,
    PSD_CALCULATION_PARAMS,
    clean_raw_eeg,
    compute_psd_from_raw,
)

mne.set_log_level('WARNING')



# ---------------------------------------------------------------------
# Frequency grid aligned to utils Welch PSD bins (no interpolation)
# ---------------------------------------------------------------------
_NFFT = int(PSD_CALCULATION_PARAMS.get("n_fft", 256))
_SFREQ = float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0))
_f = np.linspace(0.0, _SFREQ / 2.0, _NFFT // 2 + 1, dtype=float)  # Hz
_w = 2 * np.pi * _f                                                # rad s^-1


# ---------------------------------------------------------------------
# Linearized JR transfer functions and PSD
# ---------------------------------------------------------------------

def _He(jw: complex, A: float, a: float) -> complex:
    """Second-order excitatory synaptic kernel in frequency domain.

    He(s) = A*a / (s^2 + 2*a*s + a^2); here s = j*w.
    """
    return (A * a) / ((- (jw ** 2)) + 2j * a * jw + a ** 2)


def _Hi(jw: complex, B: float, b: float) -> complex:
    """Second-order inhibitory synaptic kernel in frequency domain.

    Hi(s) = B*b / (s^2 + 2*b*s + b^2); here s = j*w.
    """
    return (B * b) / ((- (jw ** 2)) + 2j * b * jw + b ** 2)


def _jr_transfer(jw: complex, p: dict[str, float]) -> complex:
    """Return linearized JR transfer T(jw) = V(jw) / P(jw).

    Using standard JR topology with linearized sigmoid slope ``G`` and
    connectivities C1..C4. With pyramidal output V being y1. The derived
    small-signal transfer (assuming S'(v0) = G) is:

        T = He^2 * G / (1 - He^2 * G^2 * C1*C2 + He*Hi * G^2 * C3*C4)

    All parameters are scalars and bundled in ``p``.
    """
    He = _He(jw, p['A'], p['a'])
    Hi = _Hi(jw, p['B'], p['b'])
    G = p['G']
    # Derived connectivities from base C1
    C1 = p['C1']
    C2 = 0.8 * C1
    C3 = 0.25 * C1
    C4 = 0.25 * C1
    denom = 1.0 - (He ** 2) * (G ** 2) * C1 * C2 + (He * Hi) * (G ** 2) * C3 * C4
    return (He ** 2) * G / denom


def _P_omega(p: dict[str, float]) -> np.ndarray:
    """Model power P(ω) on fixed grid ``_w`` (arbitrary units).

    For a white-noise external drive with flat spectrum, the output spectrum
    is |T(jw)|^2 up to a constant scaling (absorbed by the fitted ``gain``).
    """
    T = np.zeros_like(_w, dtype=np.complex128)
    for i, omega in enumerate(_w):
        jw = 1j * omega
        T[i] = _jr_transfer(jw, p)
    return (np.abs(T) ** 2).astype(float)


# ---------------------------------------------------------------------
# PSD estimation helper (MNE Raw)
# ---------------------------------------------------------------------

def compute_psd(
    raw: mne.io.BaseRaw,
    *,
    channel: str | list[str] = 'O1..',
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_fft: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, mean_psd) for *raw*.

    This matches the helper in the CTM module for a consistent interface.
    """
    raw_copy = raw.copy().pick([channel] if isinstance(channel, str) else channel)
    spectrum = raw_copy.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft)
    psds = spectrum.get_data()
    freqs = spectrum.freqs
    mean_psd = psds.mean(axis=0)
    return freqs, mean_psd


# ---------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Reduced 6‑parameter JR: single connectivity scalar C1; others derived
# C2 = 0.8*C1, C3 = 0.25*C1, C4 = 0.25*C1
# ---------------------------------------------------------------------
_PARAM_KEYS = [
    'C1',      # base connectivity; C2, C3, C4 are derived
    'A', 'B',  # synaptic gains
    'a', 'b',  # synaptic time constants (s⁻¹)
    'G',       # linearised sigmoid slope
]


def _dict_to_vector(p: dict[str, float]):
    import numpy as _np
    return _np.asarray([p[k] for k in _PARAM_KEYS], dtype=_np.float32)


_DEFAULT_THETA0 = np.asarray([
    135.0,        # C1 (C2=0.8*C1, C3=C1/4, C4=C1/4)
    3.25, 22.0,   # A, B
    100.0, 50.0,  # a, b
    1.5,          # G (effective slope)
], dtype=float)

_DEFAULT_BOUNDS = np.asarray([
    (50.0, 300.0),   # C1
    (1.0, 10.0),     # A
    (5.0, 60.0),     # B
    (50.0, 150.0),   # a
    (20.0, 120.0),   # b
    (0.1, 5.0),      # G
], dtype=float)


def _loss_function(
    theta: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
    *,
    normalize: bool = True,
    normalization: str = 'mean',
) -> float:
    """MSE between model and empirical PSD on identical Welch bins.

    The JR model is evaluated on the shared grid `_f` that matches utils' bins,
    so no interpolation is required.
    """
    p_full = dict(zip(_PARAM_KEYS, theta))
    model_psd = _P_omega(p_full)

    # Restrict comparison to configured band (e.g., up to 45 Hz)
    from utils.util import PSD_CALCULATION_PARAMS  # local to avoid circulars
    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", _SFREQ / 2.0))
    mask = (_f >= fmin) & (_f <= fmax)
    model_psd = model_psd[mask]
    real_psd = real_psd[mask]

    if normalize:
        model_psd = normalize_psd(model_psd)
        real_psd = normalize_psd(real_psd)
    return float(np.mean((model_psd - real_psd) ** 2))


class LossFunction:
    def __init__(self, freqs: np.ndarray, psd: np.ndarray, *, normalize: bool = True, normalization: str = 'mean'):
        self.freqs = freqs
        self.psd = psd
        self.normalize = normalize
        self.normalization = normalization

    def __call__(self, x):
        return _loss_function(
            np.asarray(x, dtype=float), self.freqs, self.psd,
            normalize=self.normalize, normalization=self.normalization,
        )


def fit_parameters(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    initial_theta: np.ndarray | None = None,
    sigma0: float = 0.5,
    bounds: np.ndarray | None = None,
    cma_opts: dict | None = None,
    return_full: bool = False,
    normalize: bool = True,
    normalization: str = 'mean',
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, float]:
    """Fit JR parameters to a power spectrum using CMA-ES."""
    theta0 = _DEFAULT_THETA0 if initial_theta is None else np.asarray(initial_theta, dtype=float)
    bounds_arr = _DEFAULT_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (len(_PARAM_KEYS), 2):
        raise ValueError(f'bounds must have shape ({len(_PARAM_KEYS)}, 2)')

    lower_bounds, upper_bounds = bounds_arr[:, 0], bounds_arr[:, 1]
    opts = {
        'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
        'verbose': -9,
        'verb_log': 0,
    }
    opts.setdefault('tolfun', 1e-7)
    opts.setdefault('maxiter', 600)
    if cma_opts:
        opts.update(cma_opts)

    es = cma.CMAEvolutionStrategy(theta0.tolist(), sigma0, opts)
    es.optimize(LossFunction(freqs, psd, normalize=normalize, normalization=normalization), n_jobs=-1)
    theta_best = np.asarray(es.result.xbest, dtype=float)
    best_params = dict(zip(_PARAM_KEYS, theta_best))

    if return_full:
        return best_params, theta_best, float(es.result.fbest)
    return best_params


def fit_jr_average_from_raw(
    raw: mne.io.BaseRaw,
    **fit_kwargs,
) -> 'np.ndarray | dict[str, float]':
    """Estimate average PSD via utils and fit JR model on shared Welch bins."""
    psd, freqs = compute_psd_from_raw(raw, calculate_average=True, normalize=False, return_freqs=True)
    params = fit_parameters(freqs, psd, **fit_kwargs)   
    return _dict_to_vector(params)


def fit_jr_per_channel_from_raw(
    raw: mne.io.BaseRaw,
    **fit_kwargs,
) -> np.ndarray:
    """Fit JR parameters per channel and return concatenated vector."""
    psd_matrix, freqs = compute_psd_from_raw(raw, calculate_average=False, normalize=False, return_freqs=True)
    all_params: list[np.ndarray] = []
    for row in psd_matrix:
        p = fit_parameters(freqs, row, **fit_kwargs)
        all_params.append(_dict_to_vector(p))
    return np.concatenate(all_params, axis=0)


