"""Wong-Wang mean-field model parameter fitting for EEG.

Fits the reduced Wong-Wang (NMDA gating) model to empirical power spectra
via CMA-ES on a bounded unit-cube parameterisation.

Public API:
    * ``simulate_wong_wang(T, dt, params, ...)`` - Euler-Maruyama simulation.
    * ``fit_parameters / fit_wong_wang_from_raw_cma``  - CMA-ES fitting.
    * ``fit_wong_wang_average_from_raw``  - Average-PSD wrapper.
    * ``fit_wong_wang_per_channel_from_raw`` - Per-channel wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cma
import mne
import numpy as np

mne.set_log_level("WARNING")

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

from utils.util import (
    PSD_CALCULATION_PARAMS,
    compute_psd_from_array,
    compute_psd_from_raw,
    normalize_psd,
)

__all__ = [
    "WWParams",
    "simulate_wong_wang",
    "fit_wong_wang_average_from_raw",
    "fit_wong_wang_per_channel_from_raw",
    "fit_wong_wang_from_raw_cma",
    "fit_parameters",
]


# ---------------------------------------------------------------------------
# Core Wong-Wang single-node definitions
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True, fastmath=True)
def _phi_core(x: float, d: float) -> float:
    """Firing rate nonlinearity (Numba-optimised scalar version)."""
    if abs(x) < 1e-6:
        return max(1.0 / d + x / 2.0, 0.0)
    else:
        denom = 1.0 - np.exp(-d * x)
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        return max(x / denom, 0.0)


def _phi(aI_minus_b: float, d: float) -> float:
    """Wong-Wang firing nonlinearity r(I) = (aI-b) / (1 - exp(-d(aI-b)))."""
    return _phi_core(float(aI_minus_b), d)


@dataclass
class WWParams:
    """Wong-Wang model parameters."""

    J: float = 0.95
    tau_s: float = 0.1
    gamma_gain: float = 0.641
    a: float = 270.0
    b: float = 108.0
    d: float = 0.154
    I0: float = 0.32
    sigma: float = 0.01


def simulate_wong_wang(
    T: float,
    dt: float,
    params: WWParams,
    s0: float = 0.0,
    burn_in: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate single-node Wong-Wang NMDA gating variable S(t).

    Uses Euler-Maruyama integration with clipping to [0, 1].

    Args:
        T: Simulation duration in seconds (after burn-in).
        dt: Integration time step in seconds.
        params: Model parameters.
        s0: Initial value of S.
        burn_in: Burn-in duration in seconds (discarded).
        seed: RNG seed for reproducibility.

    Returns:
        1-D array of S(t) values after burn-in.
    """
    rng = np.random.default_rng(seed)
    n_total = int(np.ceil((T + burn_in) / dt))
    S = np.empty(n_total, dtype=np.float64)
    s = float(np.clip(s0, 0.0, 1.0))

    inv_tau = 1.0 / params.tau_s
    gamma = params.gamma_gain
    a, b, d = params.a, params.b, params.d
    J, I0, sig = params.J, params.I0, params.sigma
    std = np.sqrt(dt) * sig

    for t in range(n_total):
        I = J * s + I0
        r = _phi(a * I - b, d)
        ds = (-s * inv_tau + (1.0 - s) * gamma * r) * dt + std * rng.standard_normal()
        s = np.clip(s + ds, 0.0, 1.0)
        S[t] = s

    burn_idx = int(np.floor(burn_in / dt))
    return S[burn_idx:]


# ---------------------------------------------------------------------------
# PSD utilities
# ---------------------------------------------------------------------------

_NFFT = int(PSD_CALCULATION_PARAMS.get("n_fft", 256))
_SFREQ = float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0))


def _loss_function(
    theta: np.ndarray,
    target_psd: np.ndarray,
    freqs: np.ndarray,
    *,
    sim_T: float = 10.0,
    seed: Optional[int] = None,
) -> float:
    """MSE between simulated WW log-PSD and target log-PSD."""
    J, tau_ms, gamma_gain, I0, sigma = theta
    params = WWParams(
        J=float(J),
        tau_s=float(tau_ms) / 1000.0,
        gamma_gain=float(gamma_gain),
        I0=float(I0),
        sigma=float(sigma),
    )
    y = simulate_wong_wang(T=sim_T, dt=1.0 / _SFREQ, params=params, s0=0.0, burn_in=1.0, seed=seed)
    psd_sim = compute_psd_from_array(y, sfreq=_SFREQ, normalize=False)

    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", _SFREQ / 2.0))
    mask = (freqs >= fmin) & (freqs <= fmax)

    return float(np.mean((normalize_psd(psd_sim[mask]) - normalize_psd(target_psd[mask])) ** 2))


def _ranges() -> tuple[tuple[float, float], ...]:
    """Parameter bounds for the unit-cube parameterisation."""
    return (
        (0.2, 1.6),      # J
        (40.0, 140.0),   # tau_ms
        (0.4, 1.2),      # gamma_gain
        (0.2, 0.6),      # I0
        (0.003, 0.05),   # sigma
    )


def _from_unit(u: np.ndarray) -> tuple[float, float, float, float, float]:
    """Map a [0,1]^5 vector to physical parameter ranges."""
    (J_lo, J_hi), (t_lo, t_hi), (g_lo, g_hi), (I_lo, I_hi), (s_lo, s_hi) = _ranges()
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return (
        float(J_lo + u[0] * (J_hi - J_lo)),
        float(t_lo + u[1] * (t_hi - t_lo)),
        float(g_lo + u[2] * (g_hi - g_lo)),
        float(I_lo + u[3] * (I_hi - I_lo)),
        float(s_lo + u[4] * (s_hi - s_lo)),
    )


class _WongWangLossFunction:
    """Pickleable CMA-ES objective for Wong-Wang fitting."""

    def __init__(
        self,
        target_psd: np.ndarray,
        freqs: np.ndarray,
        sim_T: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.target_psd = target_psd
        self.freqs = freqs
        self.sim_T = sim_T
        self.seed = seed

    def __call__(self, u: np.ndarray) -> float:
        J, tau_ms, gamma_gain, I0, sigma = _from_unit(u)
        theta = np.asarray([J, tau_ms, gamma_gain, I0, sigma], dtype=float)
        return _loss_function(theta, self.target_psd, self.freqs, sim_T=self.sim_T, seed=self.seed)


def fit_wong_wang_from_raw_cma(
    target_psd: np.ndarray,
    freqs: np.ndarray,
    *,
    sim_T: float = 10.0,
    seed: Optional[int] = 0,
    popsize: int = 12,
    sigma0: float = 0.2,
    max_iter: Optional[int] = None,
) -> np.ndarray:
    """Fit Wong-Wang parameters with CMA-ES on a bounded unit cube.

    Args:
        target_psd: Target PSD array.
        freqs: Frequency array matching *target_psd*.
        sim_T: Simulation length in seconds.
        seed: RNG seed.
        popsize: CMA-ES population size.
        sigma0: Initial CMA-ES spread.
        max_iter: Maximum CMA-ES iterations.

    Returns:
        Float32 parameter vector ``[J, tau_s_ms, gamma_gain, I0, sigma]``.
    """
    loss_function = _WongWangLossFunction(target_psd, freqs, sim_T, seed)

    x0 = 0.5 * np.ones(5, dtype=float)
    opts: dict = {
        "bounds": [0.0, 1.0],
        "popsize": int(popsize),
        "verbose": -9,
        "verb_log": 0,
        "tolfun": 1e-4,
        "maxiter": 600 if max_iter is None else int(max_iter),
        "seed": 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(loss_function, iterations=opts["maxiter"], n_jobs=1)

    u_best = es.result.xbest
    J, tau_ms, gamma_gain, I0, sigma = _from_unit(u_best)
    return np.asarray([J, tau_ms, gamma_gain, I0, sigma], dtype=np.float32)


# Alias for backwards compatibility
fit_parameters = fit_wong_wang_from_raw_cma


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def fit_wong_wang_average_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit Wong-Wang to the channel-averaged PSD.

    Args:
        raw: MNE Raw object.
        **fit_kwargs: Forwarded to ``fit_wong_wang_from_raw_cma``.

    Returns:
        Float32 parameter vector ``[J, tau_s_ms, gamma_gain, I0, sigma]``.
    """
    psd, freqs = compute_psd_from_raw(
        raw, calculate_average=True, normalize=False, return_freqs=True,
    )
    return fit_wong_wang_from_raw_cma(psd, freqs, **fit_kwargs)


def fit_wong_wang_per_channel_from_raw(
    raw: mne.io.BaseRaw, **fit_kwargs: object
) -> np.ndarray:
    """Fit Wong-Wang per channel and return a concatenated parameter vector.

    Args:
        raw: MNE Raw object.
        **fit_kwargs: Forwarded to ``fit_wong_wang_from_raw_cma``.

    Returns:
        Flattened float32 array of all per-channel parameter vectors.
    """
    psd_matrix, freqs = compute_psd_from_raw(
        raw, calculate_average=False, normalize=False, return_freqs=True,
    )
    out = [fit_wong_wang_from_raw_cma(psd, freqs, **fit_kwargs) for psd in psd_matrix]
    return np.array(out, dtype=np.float32).flatten()
