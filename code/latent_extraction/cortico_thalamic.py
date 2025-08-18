"""ctm_fitting.py

Refactored Cortico-Thalamic Model (CTM) parameter-fitting utilities.

The public API exposes three convenience functions:

    * compute_psd(raw, fmin=1.0, fmax=40.0, n_fft=128)
        – Returns (freqs, mean_psd) from an MNE Raw instance.

    * fit_parameters(freqs, psd, *, initial_theta=None, sigma0=0.5,
                     bounds=None, cma_opts=None, return_full=False)
        – Runs CMA-ES to fit CTM parameters to the supplied power
          spectrum.  Returns a dict of best-fit parameters.  If
          ``return_full=True`` also returns (theta_best, loss_best).

    * fit_ctm_from_raw(raw, channel='O1..', **kwargs)
        – One-liner that combines the above two steps: picks
          ``channel`` from *raw*, band-pass filters it, estimates the
          PSD, and fits the CTM.  All keyword arguments are forwarded
          to :func:`fit_parameters`.
"""

from __future__ import annotations

import numpy as np
import cma
from scipy.interpolate import interp1d
import mne
import matplotlib.pyplot as plt
import os
import re
from utils.util import compute_psd_from_raw, normalize_psd, PSD_CALCULATION_PARAMS
mne.set_log_level('WARNING')  # Suppress MNE warnings

# ---------------------------------------------------------------------
# Model constants (do not change – see Table 2 in the original paper)
# ---------------------------------------------------------------------
Lx = Ly = 0.5               # metres
k0 = 10.0                   # m^-1
gamma_e = 116.0             # s^-1
r_e = 0.086                 # metres (86 mm)

# ---------------------------------------------------------------------
# Frequency and spatial grids (global, reused across calls)
# ---------------------------------------------------------------------
# Align model evaluation frequencies to utils PSD parameters (no interpolation)
_NFFT = int(PSD_CALCULATION_PARAMS.get("n_fft", 256))
_SFREQ = float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0))
_f = np.linspace(0.0, _SFREQ / 2.0, _NFFT // 2 + 1, dtype=float)  # Hz
_w = 2 * np.pi * _f                                                # rad s^-1

_M = 10
_m = _n = np.arange(-_M, _M + 1)
_kx = 2 * np.pi * _m[:, None] / Lx
_ky = 2 * np.pi * _n[None, :] / Ly
_k2 = _kx**2 + _ky**2                  # shape (2M+1, 2M+1)
_Δk = (2 * np.pi / Lx) * (2 * np.pi / Ly)


# ---------------------------------------------------------------------
# Core CTM helpers – unchanged mathematically from the original script
# ---------------------------------------------------------------------

def _L_matrix(omega: float, alpha: float, beta: float) -> complex:
    """Second-order synaptic response function *L(ω)*."""
    return 1 / ((1 - 1j * omega / alpha) * (1 - 1j * omega / beta))


def _q2_re2(omega: float, p: dict[str, float]) -> float:
    """Compute *q² rₑ²* (real part only)."""
    Lw = _L_matrix(omega, p['alpha'], p['beta'])
    num = (1 - 1j * omega / gamma_e)**2 - 1
    den = 1 - p['G_ei'] * Lw
    bracket = (
        Lw * p['G_ee']
        + (Lw**2 * p['G_ese'] + Lw**3 * p['G_esre'])
        * np.exp(1j * omega * p['t0'])
        / (1 - Lw**2 * p['G_srs'])
    )
    return (num - bracket / den).real


def _P_omega(p: dict[str, float]) -> np.ndarray:
    """Return model power *P(ω)* on the fixed grid *_w* (arbitrary units)."""
    P = np.zeros_like(_w, dtype=float)
    for idx, omega in enumerate(_w):
        Lw = _L_matrix(omega, p['alpha'], p['beta'])
        q2 = _q2_re2(omega, p)
        denom = (
            (1 - p['G_srs'] * Lw**2)
            * (1 - p['G_ei'] * Lw)
            * (_k2 * r_e**2 + q2 * r_e**2)
        )
        phi = p['G_ese'] * np.exp(1j * omega * p['t0'] / 2) / denom
        Fk = np.exp(-_k2 / k0**2)
        P[idx] = np.sum(np.abs(phi)**2 * Fk)
    return P * _Δk




# ---------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------
# Public order of parameters (9-element vector)
_PARAM_KEYS = [
    'G_ee', 'G_ei', 'G_ese', 'G_esre', 'G_srs',
    'alpha', 'beta', 't0'
]

# ---------------------------------------------------------------------
# Helper: convert dict → ordered NumPy vector
# ---------------------------------------------------------------------

def _dict_to_vector(p: dict[str, float]):
    """Return NumPy array in canonical _PARAM_KEYS order."""
    import numpy as _np  # local import to avoid unconditional dependency
    return _np.asarray([p[k] for k in _PARAM_KEYS], dtype=_np.float32)

_DEFAULT_THETA0 = np.asarray([10.3, -11.2, 1.7, -2.7, -0.13, 58.0, 305.0, 0.08])
_DEFAULT_BOUNDS = np.asarray(
    [
        (0, 30),      # G_ee
        (-30, 0),     # G_ei
        (0, 10),      # G_ese
        (-10, 0),     # G_esre
        (-1, 0),      # G_srs
        (10, 100),    # alpha
        (100, 400),   # beta
        (0.01, 0.2),  # t0
    ],
    dtype=float,
)


def _loss_function(
    theta: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
    *,
    normalize: bool = True,
    normalization: str = 'mean',
) -> float:
    """Weighted MSE in log-space between model and empirical PSD."""
    
    model_psd = _P_omega(theta)

    log_model = normalize_psd(model_psd)
    
    log_real = normalize_psd(real_psd)

    return np.mean((log_model - log_real) ** 2)


class LossFunction:
    def __init__(self, freqs, psd, *, normalize: bool = True, normalization: str = 'mean'):
        self.freqs = freqs
        self.psd = psd
        self.normalize = normalize
        self.normalization = normalization

    def __call__(self, x):
        return _loss_function(
            np.asarray(x), self.freqs, self.psd,
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
    """Fit CTM parameters to a power spectrum using CMA-ES.

    Parameters
    ----------
    freqs
        1-D array of frequency points (Hz).
    psd
        1-D array of power values (linear units) corresponding to
        ``freqs``.
    initial_theta
        Optional 9-element array of starting parameters.  Defaults to
        the canonical values from the original script.
    sigma0
        Initial CMA-ES sampling spread.
    bounds
        (9, 2) array of lower/upper bounds.  Defaults to the original
        bounds.
    cma_opts
        Additional keyword arguments forwarded to
        :class:`cma.CMAEvolutionStrategy`.
    return_full
        If *True*, additionally returns *(theta_best, loss_best)*.
    gain
        Scalar amplitude factor multiplying the model PSD.

    Returns
    -------
    best_params
        Dict mapping parameter names to best-fit values.
    theta_best, loss_best
        Only if ``return_full`` is *True*.
    """
    theta0 = _DEFAULT_THETA0 if initial_theta is None else np.asarray(initial_theta, dtype=float)
    bounds_arr = _DEFAULT_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (8, 2):
        raise ValueError('bounds must have shape (8, 2)')

    lower_bounds, upper_bounds = bounds_arr[:, 0], bounds_arr[:, 1]
    opts = {
        'bounds': [lower_bounds.tolist(), upper_bounds.tolist()],
        'verbose': -9,       # suppress output completely
        'verb_log': 0        # don't write CMA log files
    }
    
    opts.setdefault('tolfun', 1e-8)  # Less strict convergence tolerance
    opts.setdefault('maxiter', 600)  # Allow more iterations
    
    if cma_opts:
        opts.update(cma_opts)
          

    es = cma.CMAEvolutionStrategy(theta0.tolist(), sigma0, opts)
    
    es.optimize(LossFunction(freqs, psd, normalize=normalize, normalization=normalization), n_jobs=-1)
    theta_best = es.result.xbest
    best_params = dict(zip(_PARAM_KEYS, theta_best))
    
    
    if return_full:
        return best_params, theta_best, es.result.fbest
    return best_params


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def fit_ctm_average_from_raw(raw: mne.io.BaseRaw,
    **fit_kwargs,
) -> 'np.ndarray | dict[str, float]':
    """High-level helper: estimate PSD and fit CTM in one call.

    All keyword arguments not recognised by this function are forwarded
    to :func:`fit_parameters`.
    """
    psd = compute_psd_from_raw(raw, calculate_average=True, normalize=False)
    total_params = []
    total_params.append(_dict_to_vector(fit_parameters(_f, psd, **fit_kwargs)))
    
    return np.array(total_params).flatten()

def fit_ctm_per_channel_from_raw(raw: mne.io.BaseRaw,
    **fit_kwargs,
) -> 'np.ndarray | dict[str, float]':
    """High-level helper: estimate PSD and fit CTM in one call.

    All keyword arguments not recognised by this function are forwarded
    to :func:`fit_parameters`.
    """
    psd = compute_psd_from_raw(raw, calculate_average=False, normalize=False)
    total_params = []
    for psd in psd:
        total_params.append(_dict_to_vector(fit_parameters(_f, psd, **fit_kwargs)))
    return np.array(total_params).flatten()