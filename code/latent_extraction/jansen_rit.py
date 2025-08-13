"""jansen_rit.py

Linearized Jansen–Rit (JR) neural mass model parameter-fitting utilities.

Public API mirrors the CTM helper for consistency:

    * compute_psd(raw, fmin=1.0, fmax=40.0, n_fft=128)
        – Returns (freqs, mean_psd) from an MNE Raw instance.

    * fit_parameters(freqs, psd, *, initial_theta=None, sigma0=0.5,
                     bounds=None, cma_opts=None, return_full=False)
        – Runs CMA-ES to fit JR parameters to the supplied power spectrum.
          Returns a dict of best-fit parameters. If ``return_full=True`` also
          returns (theta_best, loss_best).

    * fit_jr_from_raw(raw, **kwargs)
        – Convenience wrapper: estimate PSD from all EEG channels, average,
          and fit the JR model. All kwargs pass to :func:`fit_parameters`.
"""

from __future__ import annotations

import numpy as np
import cma
from scipy.interpolate import interp1d
import mne
import os
import re

mne.set_log_level('WARNING')


# ---------------------------------------------------------------------
# Fixed frequency grid used by the model evaluation
# ---------------------------------------------------------------------
_f = np.arange(0.5, 45.25, 0.25)  # Hz
_w = 2 * np.pi * _f               # rad s^-1


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
    denom = 1.0 - (He ** 2) * (G ** 2) * p['C1'] * p['C2'] + (He * Hi) * (G ** 2) * p['C3'] * p['C4']
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
    raw_copy = raw.copy().pick_channels([channel] if isinstance(channel, str) else channel)
    spectrum = raw_copy.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft)
    psds = spectrum.get_data()
    freqs = spectrum.freqs
    mean_psd = psds.mean(axis=0)
    return freqs, mean_psd


# ---------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------

_PARAM_KEYS = [
    'C1', 'C2', 'C3', 'C4',
    'A', 'B', 'a', 'b',
    'G', 'gain'
]


def _dict_to_vector(p: dict[str, float]):
    import numpy as _np
    return _np.asarray([p[k] for k in _PARAM_KEYS], dtype=_np.float32)


_DEFAULT_THETA0 = np.asarray([
    135.0, 108.0, 33.75, 33.75,   # C1..C4
    3.25, 22.0, 100.0, 50.0,      # A, B, a, b
    1.5,                          # G (effective slope)
    1.0,                          # gain (amplitude)
], dtype=float)

_DEFAULT_BOUNDS = np.asarray([
    (50.0, 300.0),   # C1
    (50.0, 300.0),   # C2
    (10.0, 120.0),   # C3
    (10.0, 120.0),   # C4
    (1.0, 10.0),     # A
    (5.0, 60.0),     # B
    (50.0, 150.0),   # a
    (20.0, 120.0),   # b
    (0.1, 5.0),      # G
    (1e-6, 1e3),     # gain
], dtype=float)


def _loss_function(
    theta: np.ndarray,
    freqs: np.ndarray,
    real_psd: np.ndarray,
    *,
    normalize: bool = True,
    normalization: str = 'mean',
) -> float:
    """Weighted MSE in log-space between model and empirical PSD."""
    p_full = dict(zip(_PARAM_KEYS, theta))
    gain = p_full.pop('gain')
    model_psd = gain * _P_omega(p_full)

    # Interpolate model to match empirical frequency grid
    interp_func = interp1d(_f, model_psd, kind='linear', bounds_error=False,
                           fill_value='extrapolate')
    model_resampled = interp_func(freqs)

    # Optional normalization to compare shapes independent of scale
    if normalize:
        eps = 1e-12
        if normalization == 'mean':
            real_psd = real_psd / (np.mean(real_psd) + eps)
            model_resampled = model_resampled / (np.mean(model_resampled) + eps)
        elif normalization == 'sum':
            real_psd = real_psd / (np.sum(real_psd) + eps)
            model_resampled = model_resampled / (np.sum(model_resampled) + eps)
        elif normalization == 'max':
            real_psd = real_psd / (np.max(real_psd) + eps)
            model_resampled = model_resampled / (np.max(model_resampled) + eps)
        else:
            raise ValueError("normalization must be one of {'mean','sum','max'}")

    log_model = np.log10(model_resampled + 1e-12)
    log_real = np.log10(real_psd + 1e-12)

    weights = 1.0 / (real_psd + 1e-12)
    return float(np.mean(weights * (log_model - log_real) ** 2))


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
    opts.setdefault('tolfun', 1e-8)
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


def fit_jr_from_raw(
    raw: mne.io.BaseRaw,
    *,
    n_fft: int = 128,
    as_vector: bool = False,
    **fit_kwargs,
) -> 'np.ndarray | dict[str, float]':
    """High-level helper: estimate PSD from EEG channels and fit JR model."""
    psds: list[np.ndarray] = []
    freqss: list[np.ndarray] = []

    zips = zip(raw.ch_names, raw.get_channel_types())
    eeg_chs = [name for name, type_ in zips if type_ == 'eeg']

    for ch in eeg_chs:
        try:
            freqs, psd = compute_psd(raw, channel=ch, n_fft=n_fft)
        except Exception as e:
            print(f"Error computing PSD for channel {ch}: {e}")
            return None
        psds.append(psd)
        freqss.append(freqs)

    psd = np.mean(psds, axis=0)
    freqs = freqss[0]

    params = fit_parameters(freqs, psd, **fit_kwargs)
    print(" ".join([f"{key}: {value:.5f}" for key, value in params.items()]))

    # ---- Save PSD comparison plot (name includes EEG task) ----
    task = "unknown"
    if hasattr(raw, "filenames") and getattr(raw, "filenames", None):
        fname = os.path.basename(raw.filenames[0])
        match = re.search(r"task-([^_]+)", fname)
        if match:
            task = match.group(1)

    model_psd = _P_omega({k: v for k, v in params.items() if k != 'gain'}) * params['gain']
    interp_func = interp1d(_f, model_psd, kind='linear', bounds_error=False, fill_value='extrapolate')
    model_resampled = interp_func(freqs)

    # Uncomment to save a figure
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, psd, label='Empirical PSD', color='blue')
    # plt.plot(freqs, model_resampled, label='Fitted JR PSD', color='red', linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density')
    # plt.title(f'JR Fitting ({task}): Empirical vs Fitted PSD')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'fitted_psd_jr_{task}.png')

    if as_vector:
        return _dict_to_vector(params)
    return params


