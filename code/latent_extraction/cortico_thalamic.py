"""Cortico-Thalamic Model (CTM) parameter-fitting utilities.

Fits the Robinson et al. cortico-thalamic model to empirical EEG power
spectra via CMA-ES optimisation.  The model PSD is computed analytically
on the same Welch frequency grid used for the empirical data.

Public API:
    * ``fit_parameters(freqs, psd, ...)`` - CMA-ES fit on a PSD.
    * ``fit_ctm_average_from_raw(raw)``   - Average-PSD convenience wrapper.
    * ``fit_ctm_per_channel_from_raw(raw)``- Per-channel convenience wrapper.
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
# Model constants (Table 2, Robinson et al.)
# ---------------------------------------------------------------------------
Lx = Ly = 0.5
k0 = 10.0
gamma_e = 116.0
r_e = 0.086

# ---------------------------------------------------------------------------
# Spatial grids (pre-computed, reused across calls)
# ---------------------------------------------------------------------------
_M = 10
_m = _n = np.arange(-_M, _M + 1)
_kx = 2 * np.pi * _m[:, None] / Lx
_ky = 2 * np.pi * _n[None, :] / Ly
_k2 = _kx**2 + _ky**2
_delta_k = (2 * np.pi / Lx) * (2 * np.pi / Ly)

_k2_re2 = _k2 * r_e**2
_Fk = np.exp(-_k2 / k0**2)
_re2 = r_e**2


# ---------------------------------------------------------------------------
# Core CTM transfer functions
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True, fastmath=True)
def _L_matrix(omega: complex, alpha: float, beta: float) -> complex:
    """Second-order synaptic response L(omega)."""
    return 1 / ((1 - 1j * omega / alpha) * (1 - 1j * omega / beta))


@numba.jit(nopython=True, cache=True, fastmath=True)
def _q2_re2_core(
    omega: complex,
    alpha: float,
    beta: float,
    G_ei: float,
    G_ee: float,
    G_ese: float,
    G_esre: float,
    G_srs: float,
    t0: float,
) -> float:
    """Compute q^2 * r_e^2 (real part only)."""
    Lw = _L_matrix(omega, alpha, beta)
    num = (1 - 1j * omega / gamma_e) ** 2 - 1
    den = 1 - G_ei * Lw
    bracket = (
        Lw * G_ee
        + (Lw**2 * G_ese + Lw**3 * G_esre)
        * np.exp(1j * omega * t0)
        / (1 - Lw**2 * G_srs)
    )
    return (num - bracket / den).real


def _q2_re2(omega: complex, p: dict[str, float]) -> float:
    """Compute q^2 * r_e^2 via the Numba core."""
    return _q2_re2_core(
        omega,
        p["alpha"], p["beta"], p["G_ei"], p["G_ee"],
        p["G_ese"], p["G_esre"], p["G_srs"], p["t0"],
    )


def _P_omega(p: dict[str, float], freqs: np.ndarray) -> np.ndarray:
    """Vectorised model power P(omega) on the given frequency grid.

    Args:
        p: CTM parameter dictionary.
        freqs: Frequency array in Hz.

    Returns:
        Model power spectrum array aligned with *freqs*.
    """
    w = 2 * np.pi * freqs
    Lw = _L_matrix(w, p["alpha"], p["beta"])
    q2 = _q2_re2(w, p)

    Lw_broad = Lw[:, None, None]
    q2_broad = q2[:, None, None]
    k2_re2_broad = _k2_re2[None, :, :]

    denom = (
        (1 - p["G_srs"] * Lw_broad**2)
        * (1 - p["G_ei"] * Lw_broad)
        * (k2_re2_broad + q2_broad)
    )

    exp_term = np.exp(1j * w[:, None, None] * p["t0"] / 2)
    phi = p["G_ese"] * (Lw_broad**2) * exp_term / denom
    P = np.sum(np.abs(phi) ** 2 * _Fk[None, :, :], axis=(1, 2))
    return P * _delta_k


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
_PARAM_KEYS: list[str] = [
    "G_ee", "G_ei", "G_ese", "G_esre", "G_srs",
    "alpha", "beta", "t0",
]


def _dict_to_vector(p: dict[str, float]) -> np.ndarray:
    """Return parameter dict as a NumPy array in canonical order."""
    return np.asarray([p[k] for k in _PARAM_KEYS], dtype=np.float32)


_DEFAULT_THETA0 = np.asarray(
    [10.0, -20.0, 5.0, -5.0, -0.5, 50.0, 300.0, 0.10]
)
_DEFAULT_BOUNDS = np.asarray(
    [
        (0, 20),        # G_ee
        (-40, 0),       # G_ei
        (0, 40),        # G_ese
        (-40, 0),       # G_esre
        (-5, 0),        # G_srs
        (10, 100),      # alpha
        (100, 800),     # beta
        (0.075, 0.14),  # t0  (seconds)
    ],
    dtype=float,
)


def _loss_function(
    theta: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
) -> float:
    """MSE in log-space between model and empirical PSD."""
    if isinstance(theta, np.ndarray):
        if theta.ndim != 1 or theta.size != len(_PARAM_KEYS):
            theta = theta.reshape(-1)
        p = {k: float(theta[i]) for i, k in enumerate(_PARAM_KEYS)}
    else:
        p = theta

    model_psd = _P_omega(p, freqs)
    return float(np.mean((normalize_psd(model_psd) - normalize_psd(real_psd)) ** 2))


class _LossFunction:
    """Pickleable CMA-ES objective wrapping ``_loss_function``."""

    def __init__(self, freqs: np.ndarray, psd: np.ndarray) -> None:
        self.freqs = freqs
        self.psd = psd

    def __call__(self, x: np.ndarray) -> float:
        return _loss_function(np.asarray(x), self.freqs, self.psd)


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
    """Fit CTM parameters to a power spectrum using CMA-ES.

    Args:
        freqs: 1-D frequency array (Hz).
        psd: 1-D power array (linear units) matching *freqs*.
        initial_theta: Optional 8-element starting vector.
        sigma0: Initial CMA-ES sampling spread.
        bounds: ``(8, 2)`` lower/upper bounds array.
        cma_opts: Extra options forwarded to ``cma.CMAEvolutionStrategy``.
        return_full: If True, also return ``(theta_best, loss_best)``.

    Returns:
        Best-fit parameter dict (and optionally raw vector + loss).
    """
    theta0 = _DEFAULT_THETA0 if initial_theta is None else np.asarray(initial_theta, dtype=float)
    bounds_arr = _DEFAULT_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if bounds_arr.shape != (8, 2):
        raise ValueError("bounds must have shape (8, 2)")

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

    theta_best = es.result.xbest
    best_params = dict(zip(_PARAM_KEYS, theta_best))

    if return_full:
        return best_params, theta_best, es.result.fbest
    return best_params


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def fit_ctm_average_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit CTM to the channel-averaged PSD.

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


def fit_ctm_per_channel_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit CTM per channel and return a concatenated parameter vector.

    Args:
        raw: MNE Raw object.
        **fit_kwargs: Forwarded to ``fit_parameters``.

    Returns:
        1-D float32 array of all per-channel parameters concatenated.
    """
    psd, freqs = compute_psd_from_raw(
        raw, calculate_average=False, normalize=False, return_freqs=True,
    )
    total_params = [
        _dict_to_vector(fit_parameters(freqs, channel_psd, **fit_kwargs))
        for channel_psd in psd
    ]
    return np.array(total_params).flatten()
