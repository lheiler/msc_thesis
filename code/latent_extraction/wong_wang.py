"""
Wong–Wang mean-field latent feature extractor for EEG.

Provides:
- WWParams dataclass
- simulate_wong_wang(): Euler–Maruyama single-node DMF
- fit_wong_wang_average_from_raw(): CMA-ES fitter on average PSD
- fit_wong_wang_per_channel_from_raw(): CMA-ES fitter per channel
- fit_wong_wang_from_raw_cma(): CMA-ES fitter (single-call)

Returns a compact parameter vector [J, tau_s_ms, gamma_gain, I0, sigma].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

import cma

from utils.util import (
    compute_psd_from_raw,
    compute_psd_from_array,
    normalize_psd,
    PSD_CALCULATION_PARAMS,
)

__all__ = [
    "WWParams",
    "simulate_wong_wang",
    "fit_wong_wang_average_from_raw",
    "fit_wong_wang_per_channel_from_raw",
    "fit_wong_wang_from_raw_cma",
]


# --------------------------------------
# Core Wong–Wang single-node definitions
# --------------------------------------

def _phi(aI_minus_b: np.ndarray, d: float) -> np.ndarray:
    """Wong–Wang firing nonlinearity.
    r(I) = (aI - b) / (1 - exp(-d (aI - b))). Safe for small/large args.
    """
    x = aI_minus_b.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    small = np.abs(x) < 1e-6
    if np.any(small):
        xs = x[small]
        out[small] = 1.0 / d + xs / 2.0
    big = ~small
    xb = x[big]
    denom = 1.0 - np.exp(-d * xb)
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    out[big] = xb / denom
    return np.maximum(out, 0.0)


@dataclass
class WWParams:
    J: float = 0.95          # local recurrent strength
    tau_s: float = 0.1       # seconds (NMDA ~100 ms)
    gamma_gain: float = 0.641
    a: float = 270.0         # n/C
    b: float = 108.0         # Hz
    d: float = 0.154         # s
    I0: float = 0.32         # nA baseline
    sigma: float = 0.01      # noise amplitude


def simulate_wong_wang(
    T: float,
    dt: float,
    params: WWParams,
    s0: float = 0.0,
    burn_in: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate single-node Wong–Wang NMDA gating variable S(t).

    dS/dt = -S/tau_s + (1 - S) * gamma * r(aI - b) + sigma * sqrt(dt) * N(0,1),
    with I = J * S + I0.
    Returns S(t) after burn-in.
    """
    rng = np.random.default_rng(seed)
    n_total = int(np.ceil((T + burn_in) / dt))
    S = np.empty(n_total, dtype=np.float64)
    s = float(np.clip(s0, 0.0, 1.0))

    inv_tau = 1.0 / params.tau_s
    gamma = params.gamma_gain
    a = params.a
    b = params.b
    d = params.d
    J = params.J
    I0 = params.I0
    sig = params.sigma
    std = np.sqrt(dt) * sig

    for t in range(n_total):
        I = J * s + I0
        r = _phi(a * I - b, d)
        ds = (-s * inv_tau + (1.0 - s) * gamma * r) * dt + std * rng.standard_normal()
        s = s + ds
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        S[t] = s

    burn_idx = int(np.floor(burn_in / dt))
    return S[burn_idx:]


# ----------------------
# PSD utilities aligned to utils
# ----------------------

_NFFT = int(PSD_CALCULATION_PARAMS.get("n_fft", 256))
_SFREQ = float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0))
_FREQS = np.linspace(0.0, _SFREQ / 2.0, _NFFT // 2 + 1, dtype=np.float64)


def _loss_function(theta: np.ndarray, target_psd: np.ndarray, *, sim_T: float = 8.0, seed: Optional[int] = None) -> float:
    """Compute MSE between simulated WW log-PSD and target log-PSD on identical Welch bins."""
    J, tau_ms, gamma_gain, I0, sigma = theta
    params = WWParams(J=float(J), tau_s=float(tau_ms) / 1000.0, gamma_gain=float(gamma_gain), I0=float(I0), sigma=float(sigma))
    y = simulate_wong_wang(T=sim_T, dt=1.0 / _SFREQ, params=params, s0=0.0, burn_in=1.0, seed=seed)
    psd_sim = compute_psd_from_array(y, sfreq=_SFREQ, n_fft=_NFFT, n_per_seg=PSD_CALCULATION_PARAMS.get("n_per_seg", _NFFT), n_overlap=PSD_CALCULATION_PARAMS.get("n_overlap", int(PSD_CALCULATION_PARAMS.get("n_per_seg", _NFFT)//2)), normalize=False)
    log_sim = normalize_psd(psd_sim)
    log_tgt = normalize_psd(target_psd)
    return float(np.mean((log_sim - log_tgt) ** 2))


def _ranges() -> Tuple[Tuple[float, float], ...]:
    return (
        (0.2, 1.6),     # J
        (60.0, 140.0),  # tau_ms
        (0.4, 1.2),     # gamma_gain
        (0.2, 0.6),     # I0
        (0.003, 0.05),  # sigma
    )


def _from_unit(u: np.ndarray) -> Tuple[float, float, float, float, float]:
    (J_lo, J_hi), (t_lo, t_hi), (g_lo, g_hi), (I_lo, I_hi), (s_lo, s_hi) = _ranges()
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    J = J_lo + u[0] * (J_hi - J_lo)
    tau = t_lo + u[1] * (t_hi - t_lo)
    gam = g_lo + u[2] * (g_hi - g_lo)
    I0 = I_lo + u[3] * (I_hi - I_lo)
    sig = s_lo + u[4] * (s_hi - s_lo)
    return float(J), float(tau), float(gam), float(I0), float(sig)


def fit_wong_wang_from_raw_cma(
    target_psd,
    *,
    sim_T: float = 8.0,
    seed: Optional[int] = 0,
    popsize: int = 12,
    sigma0: float = 0.2,
    max_iter: Optional[int] = None,
):
    """Fit Wong–Wang params with CMA-ES on a bounded, scaled space."""

    def loss_unit(u: np.ndarray) -> float:
        J, tau_ms, gamma_gain, I0, sigma = _from_unit(u)
        theta = np.asarray([J, tau_ms, gamma_gain, I0, sigma], dtype=float)
        return _loss_function(theta, target_psd, sim_T=sim_T, seed=seed)

    x0 = 0.5 * np.ones(5, dtype=float)
    opts = {"bounds": [0.0, 1.0], "popsize": int(popsize), "verb_disp": 0}
    if max_iter is not None:
        opts["maxiter"] = int(max_iter)
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        X = es.ask()
        fvals = [loss_unit(x) for x in X]
        es.tell(X, fvals)
    u_best = es.result.xbest
    J, tau_ms, gamma_gain, I0, sigma = _from_unit(u_best)
    vec = np.asarray([J, tau_ms, gamma_gain, I0, sigma], dtype=np.float32)
    return vec


# ---------------------------------------------------------------------
# Convenience wrappers aligned to utils (avg and per-channel)
# ---------------------------------------------------------------------

def fit_wong_wang_average_from_raw(raw, **fit_kwargs) -> np.ndarray:
    psd, _ = compute_psd_from_raw(raw, calculate_average=True, normalize=True, return_freqs=True)
    params = fit_wong_wang_from_raw_cma(psd, **fit_kwargs)
    keys = ["J", "tau_s_ms", "gamma_gain", "I0", "sigma"]
    vec = np.asarray([params[k] for k in keys], dtype=np.float32)
    return vec


def fit_wong_wang_per_channel_from_raw(raw, **fit_kwargs) -> np.ndarray:
    psd_matrix, _ = compute_psd_from_raw(raw, calculate_average=False, normalize=True, return_freqs=True)
    keys = ["J", "tau_s_ms", "gamma_gain", "I0", "sigma"]
    out: list[np.ndarray] = []
    for psd in psd_matrix:  # fit on each channel by constructing Raw would be expensive; fallback to avg proxy via CMA each time
        params = fit_wong_wang_from_raw_cma(psd, **fit_kwargs)
        out.append(np.asarray([params[k] for k in keys], dtype=np.float32))
    return np.array(out, dtype=np.float32).flatten()

