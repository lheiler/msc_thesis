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
_f = np.arange(0.5, 45.25, 0.25)        # Hz
_w = 2 * np.pi * _f                     # rad s^-1

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
# Public helper: compute PSD from an MNE Raw instance
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

    Parameters
    ----------
    raw
        MNE Raw object with EEG data.
    channel
        Channel name(s) to select.  Accepts wildcards (see MNE).
    l_freq, h_freq
        Band-pass filter edges (Hz).
    fmin, fmax
        PSD frequency limits (Hz).
    n_fft
        FFT length for Welch estimation.

    Notes
    -----
    A *copy* of the data is created internally; the original ``raw`` is
    left untouched.
    """
    raw_copy = raw.copy().pick_channels([channel] if isinstance(channel, str) else channel)
    spectrum = raw_copy.compute_psd(method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft)
    psds = spectrum.get_data()           # shape (n_sensors, n_freqs)
    freqs = spectrum.freqs
    mean_psd = psds.mean(axis=0)
    return freqs, mean_psd


# ---------------------------------------------------------------------
# Loss and optimisation
# ---------------------------------------------------------------------
# Public order of parameters (9-element vector)
_PARAM_KEYS = [
    'G_ee', 'G_ei', 'G_ese', 'G_esre', 'G_srs',
    'alpha', 'beta', 't0', 'gain'
]

# ---------------------------------------------------------------------
# Helper: convert dict → ordered NumPy vector
# ---------------------------------------------------------------------

def _dict_to_vector(p: dict[str, float]):
    """Return NumPy array in canonical _PARAM_KEYS order."""
    import numpy as _np  # local import to avoid unconditional dependency
    return _np.asarray([p[k] for k in _PARAM_KEYS], dtype=_np.float32)

_DEFAULT_THETA0 = np.asarray([10.3, -11.2, 1.7, -2.7, -0.13, 58.0, 305.0, 0.08, 1.0])
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
        (1e-4, 1e2),  # gain
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

    log_model = np.log10(model_resampled + 1e-10)
    log_real = np.log10(real_psd + 1e-10)

    weights = 1.0 / (real_psd + 1e-10)  # data‑driven weighting

    return np.mean(weights * (log_model - log_real) ** 2)


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
    if bounds_arr.shape != (9, 2):
        raise ValueError('bounds must have shape (9, 2)')

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


    # plot fitted psd vs empirical psd
    model_psd = _P_omega(best_params)
    interp_func = interp1d(_f, model_psd, kind='linear', bounds_error=False,
                           fill_value='extrapolate')
    #model_resampled = interp_func(freqs)
    #import matplotlib.pyplot as plt
    #plt.figure(figsize=(10, 6))
    #plt.plot(freqs, psd, label='Empirical PSD', color='blue')
    #plt.plot(freqs, model_resampled, label='Fitted CTM PSD', color='red', linestyle='--')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Power Spectral Density')
    #plt.title('CTM Fitting: Empirical vs Fitted PSD')
    #plt.legend()
    #plt.grid(True)
    #plt.show()  
    
    
    if return_full:
        return best_params, theta_best, es.result.fbest
    return best_params


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def fit_ctm_from_raw(
    raw: mne.io.BaseRaw,
    *,
    channel: str | list[str] = 'O1..',
    n_fft: int = 128,
    as_vector: bool = False,
    **fit_kwargs,
) -> 'np.ndarray | dict[str, float]':
    """High-level helper: estimate PSD and fit CTM in one call.

    All keyword arguments not recognised by this function are forwarded
    to :func:`fit_parameters`.
    """
    
    #for each channel, compute the PSD take average and then fit the parameters
    psds = []
    freqss = []
    
    zips = zip(raw.ch_names, raw.get_channel_types())
    eeg_chs = [name for name, type_ in zips if type_ == 'eeg']
    
    for channel in eeg_chs:
        try:
            freqs, psd = compute_psd(
                raw,
                channel=channel,
                n_fft=n_fft,
            )
        except Exception as e:
            print(f"Error computing PSD for channel {channel}: {e}")
            return None
            continue
        psds.append(psd)
        freqss.append(freqs)
    
    # Average the PSDs across channels
    psd = np.mean(psds, axis=0)
    freqs = freqss[0]  # Assuming all channels have the same frequency grid
        
    params = fit_parameters(freqs, psd, **fit_kwargs)
    # print first 5 decimals of each parameter but all in one line
    print(" ".join([f"{key}: {value:.5f}" for key, value in params.items()]))
    
    # ---- Save PSD comparison plot (name includes EEG task) ----
    # Attempt to infer the BIDS task from the raw file name
    task = "unknown"
    if hasattr(raw, "filenames") and raw.filenames:
        fname = os.path.basename(raw.filenames[0])
        match = re.search(r"task-([^_]+)", fname)
        if match:
            task = match.group(1)

    model_psd = _P_omega(params)
    interp_func = interp1d(_f, model_psd, kind='linear', bounds_error=False,
                           fill_value='extrapolate')
    model_resampled = interp_func(freqs)

    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, psd, label='Empirical PSD', color='blue')
    # plt.plot(freqs, model_resampled, label='Fitted CTM PSD', color='red', linestyle='--')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density')
    # plt.title(f'CTM Fitting ({task}): Empirical vs Fitted PSD')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'fitted_psd_{task}.png')
    
    if as_vector:
        return _dict_to_vector(params)
    return params
