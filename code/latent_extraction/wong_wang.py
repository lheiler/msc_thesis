"""
Wong–Wang mean-field latent feature extractor for EEG.

Provides:
- WWParams dataclass
- simulate_wong_wang(): Euler–Maruyama single-node DMF
- fit_wong_wang_from_raw(): lightweight random-search fitter
- fit_wong_wang_from_raw_cma(): CMA-ES fitter (if pycma available)
- extract_wong_wang(): wrapper that tries CMA-ES then falls back

Returns a compact parameter vector [J, tau_s_ms, gamma_gain, I0, sigma].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Optional dependency for MNE Raw.compute_psd at runtime
try:  # pragma: no cover
    import mne  # type: ignore
except Exception:  # pragma: no cover
    mne = None  # type: ignore

# Optional CMA-ES optimizer
try:  # pragma: no cover
    import cma  # type: ignore
except Exception:  # pragma: no cover
    cma = None  # type: ignore

__all__ = [
    "WWParams",
    "simulate_wong_wang",
    "fit_wong_wang_from_raw",
    "fit_wong_wang_from_raw_cma",
    "extract_wong_wang",
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
# PSD utilities
# ----------------------

def _welch_psd(y: np.ndarray, sfreq: float, n_per_seg: int, n_fft: int,
               fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple Welch PSD using numpy FFT. Returns (psd, freqs)."""
    step = max(1, n_per_seg // 2)
    n = len(y)
    if n < n_per_seg:
        y = np.pad(y, (0, n_per_seg - n), mode="constant")
        n = len(y)
    windows = []
    for start in range(0, n - n_per_seg + 1, step):
        seg = y[start:start + n_per_seg]
        seg = seg - np.mean(seg)
        w = np.hanning(n_per_seg)
        X = np.fft.rfft(seg * w, n=n_fft)
        Pxx = (np.abs(X) ** 2) / (np.sum(w ** 2) * sfreq)
        windows.append(Pxx)
    psd = np.mean(windows, axis=0) if windows else np.zeros(n_fft // 2 + 1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return psd[mask], freqs[mask]


# ----------------------
# Random-search fitting
# ----------------------

def fit_wong_wang_from_raw(
    raw,
    *,
    fmin: float = 1.0,
    fmax: float = 30.0,
    sim_T: float = 8.0,
    psd_window_sec: float = 2.0,
    n_iter: int = 80,
    seed: Optional[int] = 0,
    as_vector: bool = True,
):
    """Random-search fit to channel-averaged EEG log-PSD in [fmin,fmax]."""
    sfreq = float(getattr(raw.info, "sfreq", 128.0))
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                          n_per_seg=int(psd_window_sec * sfreq),
                          n_fft=int(2 ** np.ceil(np.log2(psd_window_sec * sfreq))),
                          verbose="ERROR")
    psds, freqs = psd.get_data(return_freqs=True)
    target = np.mean(psds, axis=0)
    target = np.log10(np.maximum(target, 1e-12))

    n_per_seg = int(psd_window_sec * sfreq)
    n_fft = int(2 ** np.ceil(np.log2(n_per_seg)))

    rng = np.random.default_rng(seed)
    best_loss = np.inf
    best = None

    J_range = (0.2, 1.6)
    tau_range_ms = (60.0, 140.0)
    gamma_range = (0.4, 1.2)
    I0_range = (0.2, 0.6)
    sigma_range = (0.003, 0.05)

    for _ in range(int(n_iter)):
        J = rng.uniform(*J_range)
        tau_ms = rng.uniform(*tau_range_ms)
        gamma_gain = rng.uniform(*gamma_range)
        I0 = rng.uniform(*I0_range)
        sigma = rng.uniform(*sigma_range)
        params = WWParams(J=J, tau_s=tau_ms / 1000.0, gamma_gain=gamma_gain, I0=I0, sigma=sigma)
        y = simulate_wong_wang(T=sim_T, dt=1.0 / sfreq, params=params,
                               s0=0.0, burn_in=1.0, seed=rng.integers(1e9))
        psd_sim, f_sim = _welch_psd(y, sfreq=sfreq, n_per_seg=n_per_seg,
                                    n_fft=n_fft, fmin=fmin, fmax=fmax)
        log_sim = np.log10(np.maximum(psd_sim, 1e-12))
        if log_sim.shape != target.shape:
            log_sim = np.interp(freqs, f_sim, log_sim, left=log_sim[0], right=log_sim[-1])
        loss = float(np.mean((log_sim - target) ** 2))
        if loss < best_loss:
            best_loss = loss
            best = (J, tau_ms, gamma_gain, I0, sigma)

    assert best is not None
    vec = np.asarray(best, dtype=np.float32)
    if as_vector:
        return vec
    return {
        "J": float(vec[0]),
        "tau_s_ms": float(vec[1]),
        "gamma_gain": float(vec[2]),
        "I0": float(vec[3]),
        "sigma": float(vec[4]),
        "loss": float(best_loss),
    }


# ----------------------
# CMA-ES fitting
# ----------------------

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
    raw,
    *,
    fmin: float = 1.0,
    fmax: float = 30.0,
    sim_T: float = 8.0,
    psd_window_sec: float = 2.0,
    seed: Optional[int] = 0,
    popsize: int = 12,
    sigma0: float = 0.2,
    max_iter: Optional[int] = None,
    as_vector: bool = True,
):
    """Fit Wong–Wang params with CMA-ES on a bounded, scaled space."""
    if cma is None:
        raise RuntimeError("pycma is not available. Install `cma` or use the random-search fitter.")

    sfreq = float(getattr(raw.info, "sfreq", 128.0))
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                          n_per_seg=int(psd_window_sec * sfreq),
                          n_fft=int(2 ** np.ceil(np.log2(psd_window_sec * sfreq))),
                          verbose="ERROR")
    psds, freqs = psd.get_data(return_freqs=True)
    target_log = np.log10(np.maximum(np.mean(psds, axis=0), 1e-12))

    n_per_seg = int(psd_window_sec * sfreq)
    n_fft = int(2 ** np.ceil(np.log2(n_per_seg)))

    def loss_unit(u: np.ndarray) -> float:
        J, tau_ms, gamma_gain, I0, sigma = _from_unit(u)
        params = WWParams(J=J, tau_s=tau_ms / 1000.0, gamma_gain=gamma_gain, I0=I0, sigma=sigma)
        y = simulate_wong_wang(T=sim_T, dt=1.0 / sfreq, params=params, s0=0.0, burn_in=1.0, seed=seed)
        psd_sim, f_sim = _welch_psd(y, sfreq=sfreq, n_per_seg=n_per_seg, n_fft=n_fft, fmin=fmin, fmax=fmax)
        log_sim = np.log10(np.maximum(psd_sim, 1e-12))
        if log_sim.shape != target_log.shape:
            log_sim = np.interp(freqs, f_sim, log_sim, left=log_sim[0], right=log_sim[-1])
        return float(np.mean((log_sim - target_log) ** 2))

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
    if as_vector:
        return vec
    return {
        "J": float(vec[0]),
        "tau_s_ms": float(vec[1]),
        "gamma_gain": float(vec[2]),
        "I0": float(vec[3]),
        "sigma": float(vec[4]),
        "loss": float(es.result.fbest),
    }


# ----------------------
# Public wrapper
# ----------------------

def extract_wong_wang(raw, *, use_cma: bool = True) -> np.ndarray:
    """Return float32 vector [J, tau_s_ms, gamma_gain, I0, sigma].

    If `use_cma` and `pycma` is available, use CMA-ES; otherwise fall back
    to random-search fitting.
    """
    if use_cma and cma is not None:
        try:
            return fit_wong_wang_from_raw_cma(raw, as_vector=True)
        except Exception:
            pass
    return fit_wong_wang_from_raw(raw, as_vector=True)
