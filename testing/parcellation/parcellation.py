"""

Core Pipeline
-------------
1. **Parcellation** – fetch the Schaefer 2018 atlas (default 100 parcels) 
2. **Network Dynamics** – integrate a Jansen–Rit model *per parcel*
   long-range coupling via a weight matrix `W`.
3. **Forward Model** – build a lead-field that maps parcel activity
   to EEG sensors 
4. **Fitting** – psd loss between simulated and empirical EEG and optimise model
"""


from __future__ import annotations
from scipy.fft import next_fast_len
from dataclasses import dataclass
from typing import Callable, Tuple
from mne.filter import resample
import numpy as np
import numba
from nilearn import datasets
from nilearn import plotting
from scipy.ndimage import map_coordinates
from scipy.spatial import distance_matrix
 # ------------------------------------------------------------------
 # Toggle: prune parcels whose centroids fall outside the atlas mask
 # ------------------------------------------------------------------
PRUNE_OUT_OF_MASK = True   # set to False to keep all parcels
import mne
from mne import create_info
from typing import Callable, Tuple
from mne.datasets import eegbci
import matplotlib.pyplot as plt 
from enigmatoolbox.datasets import load_sc
import nibabel as nib
from nibabel.affines import apply_affine
from collections import defaultdict
import parcel_grouping as pg
import time
from scipy.signal import welch

# ---------------- Amortized regression (PyTorch) ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


###############################################################################
# 0. PARCEL GROUPING UTILITIES
###############################################################################




def make_param_arrays(n_parcels: int,
                      group_labels: np.ndarray,
                      per_group_params: dict[int, "JRParams"],
                      default_params: "JRParams"
                      ) -> dict[str, np.ndarray]:
    """
    Return a dict of per‑parcel parameter arrays so each parcel/group can
    have its own Jansen–Rit parameters.
    """
    param_arrays = {
        field: np.full(n_parcels,
                       getattr(default_params, field),
                       dtype=np.float32)
        for field in JRParams.__annotations__.keys()
    }
    for g, grp_par in per_group_params.items():
        mask = group_labels == g
        for field in JRParams.__annotations__.keys():
            param_arrays[field][mask] = getattr(grp_par, field)
    return param_arrays

###############################################################################
# 1. PARCELLATION UTILITIES
###############################################################################

def load_parcellation(n_parcels: int = 100):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=7)
    atlas_img = nib.load(atlas.maps)
    affine = atlas_img.affine
    data = atlas_img.get_fdata().astype(int)
    labels = atlas.labels[1:]
    centroids = np.empty((n_parcels, 3))
    atlas_data = data  # alias
    for lab in range(1, n_parcels + 1):
        ijk = np.argwhere(atlas_data == lab)
        if ijk.size == 0:
            raise RuntimeError(f"Label {lab} not found in atlas data")

        # initial centroid in ijk and xyz
        centroid_ijk_f = ijk.mean(axis=0)
        xyz = apply_affine(affine, centroid_ijk_f)

        # if that xyz lies outside the mask (i.e., trilinear interp → 0), snap
        ijk_int = np.round(centroid_ijk_f).astype(int)
        if atlas_data[tuple(ijk_int)] == 0:
            # search neighbourhood radius 2 voxels for any voxel with this label
            found = False
            for radius in (0,):
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            cand = ijk_int + np.array([dx, dy, dz])
                            if (0 <= cand[0] < atlas_data.shape[0] and
                                0 <= cand[1] < atlas_data.shape[1] and
                                0 <= cand[2] < atlas_data.shape[2] and
                                atlas_data[tuple(cand)] == lab):
                                ijk_int = cand
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            # if still not found, fall back to original fractional coordinate
        xyz = apply_affine(affine, ijk_int)
        centroids[lab - 1] = xyz
    return atlas_img, labels, centroids

# ------------------------------------------------------------------
#   Diagnostic: check and visualise parcel centroids
# ------------------------------------------------------------------
def sanity_check_centroids(centroids: np.ndarray, atlas_img):
    """Print distance stats, verify inside-mask, launch html viewer."""
    atlas_data = atlas_img.get_fdata()
    aff        = atlas_img.affine
    # voxel coords
    ijk_coords = np.linalg.inv(aff) @ np.c_[centroids, np.ones(len(centroids))].T
    ijk_coords = ijk_coords[:3].T
    # atlas value at each centroid (trilinear interp)
    values = map_coordinates(atlas_data, ijk_coords.T, order=1)
    outside = np.where(values <= 0)[0]
    if len(outside):
        print("⚠️  Centroids outside atlas mask:", outside)
    else:
        print("✅ All centroids lie inside atlas mask.")

    # distance stats
    D = distance_matrix(centroids, centroids)
    np.fill_diagonal(D, np.inf)
    print(f"Median NN‑distance: {np.median(D.min(axis=1)):.1f} mm  (min {D.min():.1f} mm)")

    # interactive view (support nilearn<=0.9 and >=0.10)
    # try:
    #     try:  # new nilearn >=0.10
    #         view = plotting.view_markers(marker_coords=centroids, marker_size=4,
    #                                      marker_color='auto',
    #                                      title="Schaefer centroids (MNI)")
    #     except TypeError:
    #         # fallback for older nilearn
    #         view = plotting.view_markers(coords=centroids, marker_size=4,
    #                                      marker_color='auto',
    #                                      title="Schaefer centroids (MNI)")
    #     view.open_in_browser()
    # except Exception as err:
    #     print("Could not launch html viewer:", err)

def _canon(name: str) -> str:
    """Return an *uppercase* channel key stripped of EEG‑BCI boilerplate."""
    name = name.upper()
    for trash in ("EEG ", "-REF", "REF", "-LE", ":", "."):
        name = name.replace(trash, "")
    return name.strip()

###############################################################################
###############################################################################
# 2. JANSEN–RIT MODEL
###############################################################################

DEFAULT_COUPLING_GAIN = 0.15

@dataclass(frozen=True)
class JRParams:
    A: float = 3.25
    B: float = 22.0
    a: float = 100.0
    b: float = 50.0
    C: float = 135.0
    p_mean: float = 120.0
    p_std: float = 30.0
    v0: float = 6.0
    e0: float = 2.5
    r: float = 0.56

    def sigmoid(self, v: np.ndarray) -> np.ndarray:
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - v)))


@numba.njit
def jr_rhs(t: float, y: np.ndarray,
           n: int, W: np.ndarray,
           param_A, param_B, param_a, param_b, param_C, param_v0, param_e0, param_r,
           p_ext: np.ndarray) -> np.ndarray:
    """
    Vectorised Jansen–Rit RHS where every parcel may have distinct parameters.
    """
    A, B, a, b, C = param_A, param_B, param_a, param_b, param_C
    v0, e0, r = param_v0, param_e0, param_r

    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C

    y = y.reshape(n, 6)
    x1, x2, x3, z1, z2, z3 = y.T

    sigmoid_c1x1 = 2.0 * e0 / (1.0 + np.exp(r * (v0 - C1 * x1)))
    sigmoid_x2_x3 = 2.0 * e0 / (1.0 + np.exp(r * (v0 - (x2 - x3))))
    sigmoid_c2x1 = 2.0 * e0 / (1.0 + np.exp(r * (v0 - C1 * x1)))
    sigmoid_c3x1 = 2.0 * e0 / (1.0 + np.exp(r * (v0 - C3 * x1)))

    # Ensure sigmoid_c1x1 is float32 for matmul with W (float32)
    coupling = W @ sigmoid_c1x1.astype(np.float32)

    dx1 = z1
    dz1 = A * a * sigmoid_x2_x3 - 2 * a * z1 - a**2 * x1

    dx2 = z2
    dz2 = A * a * (p_ext + C2 * sigmoid_c2x1 + coupling) - 2 * a * z2 - a**2 * x2

    dx3 = z3
    dz3 = B * b * (C4 * sigmoid_c3x1) - 2 * b * z3 - b**2 * x3

    out = np.empty((n, 6), dtype=np.float32)
    out[:, 0] = dx1.astype(np.float32)
    out[:, 1] = dx2.astype(np.float32)
    out[:, 2] = dx3.astype(np.float32)
    out[:, 3] = dz1.astype(np.float32)
    out[:, 4] = dz2.astype(np.float32)
    out[:, 5] = dz3.astype(np.float32)
    return out.ravel()


Drift      = Callable[[float, np.ndarray, Tuple], np.ndarray]
Diffusion  = Callable[[float, np.ndarray, Tuple], np.ndarray]

@numba.njit
def euler_maruyama_jit(
    f,
    sigma: float,
    y0: np.ndarray,
    t_eval: np.ndarray,
    args,
    noise: np.ndarray,
    p_drive: np.ndarray,
) -> np.ndarray:
    n_steps = len(t_eval)
    y = np.empty((n_steps, len(y0)), dtype=np.float32)
    y[0] = y0

    for i in range(1, n_steps):
        dt = t_eval[i] - t_eval[i - 1]
        sqrt_dt = np.sqrt(dt)

        drift = f(t_eval[i - 1], y[i - 1], *args, p_drive[i - 1])
        dW = noise[i - 1] * sqrt_dt

        y[i] = y[i - 1] + drift * dt + sigma * dW

    return y

def simulate_jr_network(
    duration: float,
    dt: float,
    W: np.ndarray,
    *,
    params: JRParams | None = None,
    group_labels: np.ndarray | None = None,
    per_group_params: dict[int, JRParams] | None = None,
    coupling_gain: float = DEFAULT_COUPLING_GAIN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the Jansen–Rit network with Euler–Maruyama.

    Returns
    -------
    t : (n_times,)   – time vector
    v : (n_parcels, n_times) – pyramidal potential (µV)
    """
    if params is None:
        params = JRParams()

    n = W.shape[0]
    t_eval = np.arange(0.0, duration + dt/2, dt)
    t_eval = t_eval.astype(np.float32)
    y0 = np.zeros(n * 6, dtype=np.float32)
    W = W.astype(np.float32)

    if group_labels is None:
        group_labels = np.zeros(n, dtype=int)
    if per_group_params is None:
        per_group_params = {int(group_labels[0]): params}

    param_arrays = make_param_arrays(n, group_labels, per_group_params, params)

    # ------------------------------------------------------------------
    # Parcel‐wise heterogeneity (±5 %) on a and b
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)  # deterministic seed for fitting
    jitter = 1 + 0.05 * rng.standard_normal(n).astype(np.float32)
    param_arrays["a"] *= jitter
    param_arrays["b"] *= jitter

    # ------------------------------------------------------------------
    # Ornstein–Uhlenbeck noise on external drive p(t)
    # ------------------------------------------------------------------
    tau_ou   = 0.02                     # 50 ms correlation time
    sigma_ou = 0.4 * params.p_std       # 30 % of p_std
    alpha_ou = np.exp(-dt / tau_ou)

    ou_state = rng.normal(0, params.p_std, size=n).astype(np.float32)
    p_drive = np.empty((len(t_eval), n), dtype=np.float32)
    for k in range(len(t_eval)):
        ou_state = alpha_ou * ou_state + np.sqrt(1 - alpha_ou**2) * sigma_ou * rng.standard_normal(n)
        p_drive[k] = params.p_mean + ou_state

    print(f"Simulating Jansen–Rit network with {n} parcels for {duration:.2f} s ...")

    # Scale coupling matrix by gain (but keep asymmetry and self-loops)
    W = coupling_gain * W

    # >>> integrate
    noise = np.random.normal(0.0, 1.0, size=(len(t_eval)-1, len(y0))).astype(np.float32)
    time1 = time.time()
    sol = euler_maruyama_jit(
        f=jr_rhs,
        sigma=0.0,
        y0=y0,
        t_eval=t_eval,
        args=(n, W.astype(np.float32),
              param_arrays["A"].astype(np.float32), param_arrays["B"].astype(np.float32),
              param_arrays["a"].astype(np.float32), param_arrays["b"].astype(np.float32),
              param_arrays["C"].astype(np.float32),
              param_arrays["v0"].astype(np.float32), param_arrays["e0"].astype(np.float32),
              param_arrays["r"].astype(np.float32)),
        noise=noise,
        p_drive=p_drive,
    )
    time2 = time.time()
    print(f"Simulation took {time2 - time1:.2f} seconds.")
    #print("Simulation complete.")

    # ------------------------------------------------------------------
    # post-processing
    # ------------------------------------------------------------------
    Y = sol.T
    x1 = Y[0::6]
    x2 = Y[1::6]
    v = x1

    eeg_uV = 1e-3*v
    # eeg_uV = v
    return t_eval, eeg_uV

###############################################################################
# 3. SIMPLE EEG FORWARD MODEL
###############################################################################



def build_forward_matrix(centroids_mni: np.ndarray, *, montage_name: str = "standard_1005", pick_ch_names: list[str] | None = None, sfreq = 128.0):
    """
    Return (leadfield, info) with *robust* name‑matching.
    Applies a centre-of-mass (CoM) translation to parcel centroids to roughly align them to the head frame.
    """
    # full montage first
    full = mne.channels.make_standard_montage(montage_name)
    full_pos = full.get_positions()["ch_pos"]

    if pick_ch_names is None:
        use_names = list(full_pos.keys())
    else:
        # map requested names → canonical → real name
        rev = {_canon(ch): ch for ch in full_pos}
        use_names = []
        for req in pick_ch_names:
            key = _canon(req)
            if key in rev:
                use_names.append(rev[key])
        if not use_names:
            raise ValueError(
                "None of the requested channels matched the montage.\n"
                "First few requested: %s\nFirst few montage names: %s" % (pick_ch_names[:5], list(full_pos)[:5])
            )

    sel_pos = {ch: full_pos[ch] for ch in use_names}
    montage = mne.channels.make_dig_montage(ch_pos=sel_pos, coord_frame="head")

    info = create_info(use_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)

    n_parcels, n_sensors = centroids_mni.shape[0], len(use_names)
    leadfield = np.zeros((n_sensors, n_parcels))

    sensor_xyz = np.array([sel_pos[ch] for ch in use_names]) * 1e3  # m → mm
    # ------------------------------------------------------------------
    # Rough head‑frame alignment: translate MNI centroids so their centre
    # of mass matches the centre of the selected electrode cloud.
    # This is a pragmatic fix when no subject‑specific MRI→head transform
    # is available.
    # ------------------------------------------------------------------
    centroid_com = centroids_mni.mean(axis=0)
    sensor_com   = sensor_xyz.mean(axis=0)
    centroids_head = centroids_mni + (sensor_com - centroid_com)  # translate only

    for j in range(n_parcels):
        r = np.linalg.norm(sensor_xyz - centroids_head[j], axis=1)
        leadfield[:, j] = 1.0 / np.maximum(r, 1e-3) ** 2

    leadfield /= np.linalg.norm(leadfield, axis=0, keepdims=True)
    return leadfield, info


# ------------------------------------------------------------------
#   Diagnostic plot: sensors vs. parcel centroids in the same frame
# ------------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D projection

def plot_sensors_vs_centroids(centroids_mm: np.ndarray, sim_info):
    """3‑D scatter of EEG sensor positions (red) and parcel centroids (blue)."""
    # sensor positions are stored in metres in ch['loc'][:3]
    sensor_xyz = np.array([ch['loc'][:3] for ch in sim_info['chs']]) * 1e3  # → mm

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(sensor_xyz[:, 0], sensor_xyz[:, 1], sensor_xyz[:, 2],
               c='red', marker='o', s=20, label='EEG sensors')
    ax.scatter(centroids_mm[:, 0], centroids_mm[:, 1], centroids_mm[:, 2],
               c='blue', marker='^', s=15, label='Parcel centroids')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title('Sensors (red) vs Parcel Centroids (blue)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def parcel_to_sensor(parcel_ts: np.ndarray, leadfield: np.ndarray) -> np.ndarray:
    """Linear projection of parcel time series to sensors."""
    return leadfield @ parcel_ts

###############################################################################
# 4. FITTING UTILITIES
###############################################################################

def lf(sim: np.ndarray, real: np.ndarray, sfreq: float) -> float: # 
    nperseg = min(2**next_fast_len(min(sim.shape[1], real.shape[1])), 512)
    f_sim, psd_sim = welch(sim, fs=sfreq, axis=1, nperseg=nperseg)
    f_real, psd_real = welch(real, fs=sfreq, axis=1, nperseg=nperseg)
    # trim to common frequency grid
    min_bins = min(psd_sim.shape[1], psd_real.shape[1])
    psd_sim, psd_real = psd_sim[:, :min_bins], psd_real[:, :min_bins]

    # Normalize PSDs
    psd_sim /= psd_sim.sum(axis=1, keepdims=True)
    psd_real /= psd_real.sum(axis=1, keepdims=True)

    return np.mean((psd_sim - psd_real) ** 2)

def lf_psd(sim, real, sfreq, fmin=1., fmax=45.):
    # multitaper gives smooth PSD + useful dB units
    psd_sim, freqs = mne.time_frequency.psd_array_multitaper(
        sim, sfreq, fmin=fmin, fmax=fmax, normalization="full", verbose=False)
    psd_real, _ = mne.time_frequency.psd_array_multitaper(
        real, sfreq, fmin=fmin, fmax=fmax, normalization="full", verbose=False)

    # per‑channel L1 normalisation (shape only)
    psd_sim = psd_sim / (psd_sim.sum(axis=1, keepdims=True) + 1e-20)
    psd_real = psd_real / (psd_real.sum(axis=1, keepdims=True) + 1e-20)

    loss = np.mean((np.log(psd_sim + 1e-20) - np.log(psd_real + 1e-20))**2)
    return loss

def lf_alpha_weighted(sim, real, sfreq):
    f, P_sim = welch(sim, sfreq, axis=1, nperseg=512)
    _, P_real = welch(real, sfreq, axis=1, nperseg=512)
    # normalise
    P_sim /= P_sim.sum(axis=1, keepdims=True)
    P_real /= P_real.sum(axis=1, keepdims=True)
    #plot both psds
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, P_sim.T, color='blue', alpha=0.5, label='Simulated')
    plt.semilogy(f, P_real.T, color='red', alpha=0.5, label='Real')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (a.u.)')
    plt.title('Power Spectral Density Comparison')
    plt.legend()
    plt.show()
    w = np.ones_like(f)
    w[(f >= 8) & (f <= 13)] *= 4.0      # 4× penalty outside alpha
    w /= w.mean()                       # keep scale stable
    return np.mean(w * (P_sim - P_real)**2)

def psd_log_mse(sim, real, sfreq, nperseg=512, fmin=1, fmax=45):
    # Welch PSD for both data sets
    f_sim, P_sim = welch(sim, sfreq, axis=1, nperseg=nperseg)
    f_real, P_real = welch(real, sfreq, axis=1, nperseg=nperseg)

    # restrict frequency band
    band = (f_sim >= fmin) & (f_sim <= fmax)
    P_sim, P_real = P_sim[:, band], P_real[:, band]

    # plot a comparison of the two PSDs
    plt.figure(figsize=(10, 5))
    plt.semilogy(f_sim[band], P_sim.T, color='blue', alpha=0.5, label='Simulated')
    plt.semilogy(f_real[band], P_real.T, color='red', alpha=0.5, label='Real')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (a.u.)')
    plt.title('Power Spectral Density Comparison')          
    plt.legend()
    plt.show()

    # per‑channel L1 normalisation (shape only)
    P_sim = P_sim / (P_sim.sum(axis=1, keepdims=True) + 1e-20)
    P_real = P_real / (P_real.sum(axis=1, keepdims=True) + 1e-20)

    # log‑space MSE on normalised shapes
    loss = np.mean((np.log10(P_sim + 1e-20) - np.log10(P_real + 1e-20))**2)
    return loss

# def correlation_lf(sim, real):
#     """Compute the correlation loss between simulated and real EEG."""
#     # Ensure both have the same shape
#     min_len = min(sim.shape[1], real.shape[1])
#     sim = sim[:, :min_len]
#     real = real[:, :min_len]
    
#     corr = np.corrcoef(sim, real, rowvar=False)
#     return 1 - np.mean(corr)  # 0 is perfect match, 1 is worst match

def correlation_lf(sim, real):
    # Ensure equal length
    min_len = min(sim.shape[1], real.shape[1])
    sim, real = sim[:, :min_len], real[:, :min_len]
    # Compute correlation between flattened log PSD vectors
    f_sim, P_sim = welch(sim, sfreq, axis=1, nperseg=512)
    _,    P_real = welch(real, sfreq, axis=1, nperseg=512)
    log_sim  = np.log10(P_sim + 1e-20).ravel()
    log_real = np.log10(P_real + 1e-20).ravel()
    r = np.corrcoef(log_sim, log_real)[0, 1]

    #plot a comparison of the two log PSDs (channel‑averaged)
    n_ch, n_freq = P_sim.shape
    mean_log_sim  = log_sim.reshape(n_ch, n_freq).mean(axis=0)
    mean_log_real = log_real.reshape(n_ch, n_freq).mean(axis=0)
    # plt.figure(figsize=(10, 5))
    # plt.plot(f_sim, mean_log_sim,  color='blue', alpha=0.5, label='Simulated (mean)')
    # plt.plot(f_sim, mean_log_real, color='red',  alpha=0.5, label='Real (mean)')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Log Power Spectral Density (a.u.)')
    # plt.title('Channel‑Averaged Log PSD')
    # plt.legend()
    # plt.show()
        # --- alpha‑band weighted term -----------------------------------
    alpha_mask = (f_sim >= 8) & (f_sim <= 15)
    mse_alpha = np.mean((mean_log_sim[alpha_mask] - mean_log_real[alpha_mask])**2)
    return (1 - r) + 4.0 * mse_alpha      # 0 = perfect, 2 = worst



###############################################################################
# 4B. AMORTIZED REGRESSION (PYTORCH)
###############################################################################

JR_KEYS = ["A", "B", "a", "b", "C", "p_mean"]  # per-group outputs

# -------------------------- helpers: bounds + torch-welch --------------------------

def _sigmoid_to_bounds(z: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    # elementwise map from R → [low, high]
    s = torch.sigmoid(z)
    return lows + (highs - lows) * s


def torch_welch(x: torch.Tensor, fs: float, nperseg: int, noverlap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Welch PSD in PyTorch.
    x: [B, C, T]
    returns (freqs[Tf], P[B, C, Tf])
    """
    B, C, T = x.shape
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")
    # number of frames
    n_frames = 1 + (T - nperseg) // step
    if n_frames <= 0:
        raise ValueError("Time series shorter than nperseg")
    # frame extraction
    x_frames = x.unfold(dimension=2, size=nperseg, step=step)  # [B, C, n_frames, nperseg]
    window = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    xw = x_frames * window  # broadcast over last dim
    X = torch.fft.rfft(xw, dim=-1)
    P = (X.real**2 + X.imag**2)
    P = P.mean(dim=2)  # average over frames → [B, C, F]
    freqs = torch.fft.rfftfreq(nperseg, d=1.0/float(fs)).to(x.device)
    return freqs, P

class JRParamRegressor(nn.Module):
    """Simple MLP that maps EEG PSD features to JR parameters (per group) + g.
    Output size = G * len(JR_KEYS) + 1 (for global g).
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_psd_features(eeg: np.ndarray, sfreq: float, fmin: float = 1., fmax: float = 45., nperseg: int = 512,
                          log: bool = True, per_channel_norm: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, freqs). Features are flattened [n_ch * n_freq] and **per‑channel L1‑normalised** to enforce shape‑only comparison.
    eeg shape = (n_ch, n_times).
    """
    f, P = welch(eeg, sfreq, axis=1, nperseg=nperseg)
    band = (f >= fmin) & (f <= fmax)
    P = P[:, band]
    if per_channel_norm:
        P = P / (P.sum(axis=1, keepdims=True) + 1e-20)
    if log:
        P = np.log10(P + 1e-20)
    feats = P.astype(np.float32).ravel()
    return feats, f[band]


def _sample_group_params(G: int, bounds: dict[str, tuple[float, float]], base: JRParams) -> dict[int, JRParams]:
    """Sample JR parameters per group uniformly within bounds."""
    rng = np.random.default_rng()
    params = {}
    for g in range(G):
        upd = base.__dict__.copy()
        for k in JR_KEYS:
            lo, hi = bounds[k]
            upd[k] = float(rng.uniform(lo, hi))
        params[g] = JRParams(**upd)
    return params


def _params_to_vector(per_group_params: dict[int, JRParams], g_val: float, G: int) -> np.ndarray:
    vec = []
    for grp in range(G):
        p = per_group_params[grp]
        vec.extend([getattr(p, k) for k in JR_KEYS])
    vec.append(float(g_val))
    return np.array(vec, dtype=np.float32)


def _vector_to_params(vec: np.ndarray, G: int, base: JRParams) -> tuple[dict[int, JRParams], float]:
    vec = np.asarray(vec, dtype=np.float32)
    out = {}
    idx = 0
    for grp in range(G):
        upd = base.__dict__.copy()
        for k in JR_KEYS:
            upd[k] = float(vec[idx]); idx += 1
        out[grp] = JRParams(**upd)
    g_val = float(vec[idx])
    return out, g_val


class SyntheticJRDataset(Dataset):
    """Generate (features, target) pairs on the fly by simulating the JR network.
    Targets are parameter vectors (per-group JR keys + global g).
    """
    def __init__(self, n_samples: int, duration: float, dt: float, W: np.ndarray, leadfield: np.ndarray,
                 group_labels: np.ndarray, bounds: dict[str, tuple[float, float]], base_params: JRParams,
                 sfreq_out: float, fmin: float = 1., fmax: float = 45., seed: int = 1234, nperseg: int = 512):
        self.n = n_samples
        self.duration = duration
        self.dt = dt
        self.W = W
        self.leadfield = leadfield
        self.group_labels = group_labels
        self.bounds = bounds
        self.base = base_params
        self.sfreq_out = sfreq_out
        self.fmin = fmin
        self.fmax = fmax
        self.rng = np.random.default_rng(seed)
        self.G = int(group_labels.max()) + 1
        self.nperseg = nperseg

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # sample params per group + global coupling
        per_group = {}
        for g in range(self.G):
            upd = self.base.__dict__.copy()
            for k in JR_KEYS:
                lo, hi = self.bounds[k]
                upd[k] = float(self.rng.uniform(lo, hi))
            per_group[g] = JRParams(**upd)
        g_val = float(self.rng.uniform(self.bounds["g"][0], self.bounds["g"][1]))

        # simulate
        t, parcel_ts = simulate_jr_network(
            duration=self.duration, dt=self.dt, W=self.W,
            params=self.base, group_labels=self.group_labels,
            per_group_params=per_group, coupling_gain=g_val,
        )
        eeg = parcel_to_sensor(parcel_ts, self.leadfield)
        # resample to desired sfreq
        sim_sfreq = 1.0 / self.dt
        eeg = mne.filter.resample(eeg, down=sim_sfreq / self.sfreq_out, axis=1)
        eeg = mne.filter.filter_data(eeg, sfreq=self.sfreq_out, l_freq=self.fmin, h_freq=self.fmax, verbose=False)

        x, _ = compute_psd_features(eeg, self.sfreq_out, fmin=self.fmin, fmax=self.fmax, nperseg=self.nperseg)
        y = _params_to_vector(per_group, g_val, self.G)
        return torch.from_numpy(x), torch.from_numpy(y)



# --------------------------------------------------------------------------
# 4C. Differentiable Torch simulator for PSD-to-PSD training via simulation
# --------------------------------------------------------------------------

def _torch_param_arrays(n_parcels: int,
                        group_labels: torch.Tensor,
                        per_group_vec: torch.Tensor,
                        base: JRParams,
                        G: int) -> dict[str, torch.Tensor]:
    """Create per-parcel arrays for A,B,a,b,C,v0,e0,r from a per-group parameter vector.
    per_group_vec has shape [B, G*len(JR_KEYS)+1]; the last element is g (unused here).
    Returns a dict of tensors [B, n_parcels].
    """
    B = per_group_vec.shape[0]
    jr_keys = JR_KEYS
    out = {k: torch.full((B, n_parcels), getattr(base, k), dtype=torch.float32, device=per_group_vec.device)
           for k in ['A','B','a','b','C','p_mean','v0','e0','r']}
    # copy fixed scalars from base for v0,e0,r
    # fill trainable per-group keys from vector
    idx = 0
    for g in range(G):
        mask = (group_labels == g).unsqueeze(0).expand(B, n_parcels)  # [B, N]
        for k in jr_keys:
            vals = per_group_vec[:, idx].unsqueeze(1).expand(B, n_parcels)
            out[k][mask] = vals[mask]
            idx += 1
    return out


def torch_simulate_jr_network_psd(
    per_group_vec: torch.Tensor,
    W: torch.Tensor,
    group_labels: torch.Tensor,
    leadfield: torch.Tensor,
    *,
    base_params: JRParams,
    duration: float,
    dt: float,
    sfreq_out: float,
    fmin: float = 1., fmax: float = 45.,
    nperseg: int = 512,
    noverlap_frac: float = 0.5,
    noise_strength: float = 0.0,
    jitters: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a Torch version of the JR simulator and return **normalised log-PSD features** and frequencies.
    per_group_vec: [B, G*len(JR_KEYS)+1] (last entry per sample is global coupling g).
    Returns: (x_feats [B, n_ch*n_freq], freqs[band])
    Notes: Uses fixed OU noise (no grad) so gradients flow through dynamics.
    """
    device = per_group_vec.device
    B = per_group_vec.shape[0]
    N = W.shape[0]
    G = int(group_labels.max().item()) + 1

    # unpack g and per-group params
    theta = per_group_vec
    g_val = theta[:, -1]  # [B]
    params = _torch_param_arrays(N, group_labels, theta[:, :-1], base_params, G)

    # parcel heterogeneity ±5% on a,b (no grad)
    if jitters:
        rng = torch.Generator(device=device)
        jitter = (1.0 + 0.05 * torch.randn(N, device=device)).clamp_min(0.01)
        params['a'] = params['a'] * jitter
        params['b'] = params['b'] * jitter

    # time grid
    T = int(np.round(duration / dt)) + 1
    t = torch.linspace(0.0, duration, T, device=device, dtype=torch.float32)

    # dynamics state [B, N, 6]
    y = torch.zeros(B, N, 6, device=device, dtype=torch.float32)
    # store pyramidal potential x1 over time
    V = torch.zeros(B, N, T, device=device, dtype=torch.float32)
    V[:, :, 0] = y[..., 0]

    # OU noise for p_drive (no grad)
    tau_ou = 0.02
    sigma_ou = noise_strength * (0.4 * base_params.p_std)
    alpha_ou = torch.exp(torch.tensor(-dt / tau_ou, device=device))
    ou = torch.randn(B, N, device=device) * base_params.p_std

    # scale coupling
    Wg = W * g_val.view(B, 1, 1)  # broadcast over B

    # integration loop (Euler)
    for i in range(1, T):
        # update OU
        ou = alpha_ou * ou + torch.sqrt(1 - alpha_ou**2) * sigma_ou * torch.randn(B, N, device=device)
        p_drive = params['p_mean'] + ou  # [B, N]

        x1, x2, x3, z1, z2, z3 = y[...,0], y[...,1], y[...,2], y[...,3], y[...,4], y[...,5]
        # sigmoids
        v0 = params['v0']; e0 = params['e0']; r = params['r']; C = params['C']
        C1, C2, C3, C4 = C, 0.8*C, 0.25*C, 0.25*C
        sig_c1x1 = 2.0*e0 / (1.0 + torch.exp(r * (v0 - C1 * x1)))
        sig_x2_x3 = 2.0*e0 / (1.0 + torch.exp(r * (v0 - (x2 - x3))))
        sig_c2x1 = 2.0*e0 / (1.0 + torch.exp(r * (v0 - C1 * x1)))
        sig_c3x1 = 2.0*e0 / (1.0 + torch.exp(r * (v0 - C3 * x1)))

        # coupling: [B,N] ← [B,N,N] @ [B,N,1]
        coup = torch.matmul(Wg, sig_c1x1.unsqueeze(-1)).squeeze(-1)

        A = params['A']; Bp = params['B']; a = params['a']; b = params['b']
        dx1 = z1
        dz1 = A * a * sig_x2_x3 - 2*a*z1 - (a**2)*x1
        dx2 = z2
        dz2 = A * a * (p_drive + C2 * sig_c2x1 + coup) - 2*a*z2 - (a**2)*x2
        dx3 = z3
        dz3 = Bp * b * (C4 * sig_c3x1) - 2*b*z3 - (b**2)*x3

        y = y + dt * torch.stack([dx1, dx2, dx3, dz1, dz2, dz3], dim=-1)
        V[:, :, i] = y[..., 0]

    # project parcel time series to sensors: leadfield [C,N], V [B,N,T] → eeg [B,C,T]
    eeg = torch.einsum('cn,bnt->bct', leadfield, V)

    # resample by simple decimation to match sfreq_out if needed
    sim_sfreq = 1.0/dt
    decim = max(1, int(round(sim_sfreq / sfreq_out)))
    eeg = eeg[..., ::decim]

    # Welch PSD (differentiable)
    noverlap = int(nperseg * noverlap_frac)
    freqs, P = torch_welch(eeg, fs=sfreq_out, nperseg=nperseg, noverlap=noverlap)
    band = (freqs >= fmin) & (freqs <= fmax)
    P = P[..., band]
    # per-channel L1 normalisation and log
    P = P / (P.sum(dim=-1, keepdim=True) + 1e-20)
    P = torch.log10(P + 1e-20)
    feats = P.reshape(B, -1)
    return feats, freqs[band]


def train_amortized_via_sim(
    n_sims: int,
    duration: float,
    dt: float,
    W: np.ndarray,
    leadfield: np.ndarray,
    group_labels: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    base_params: JRParams | None,
    sfreq_out: float,
    *,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-3,
    hidden: int = 512,
    depth: int = 3,
    fmin: float = 1.,
    fmax: float = 45.,
    device: str | None = None,
):
    """Train the regressor with the loss computed by **running the simulator** on the
    predicted parameters and comparing the resulting PSD to the **input** PSD.
    This matches your requested objective exactly.
    """
    if base_params is None:
        base_params = JRParams()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticJRDataset(
        n_samples=n_sims, duration=duration, dt=dt, W=W, leadfield=leadfield,
        group_labels=group_labels, bounds=bounds, base_params=base_params,
        sfreq_out=sfreq_out, fmin=fmin, fmax=fmax,
    )
    x0, y0 = dataset[0]
    in_dim = int(x0.numel()); out_dim = int(y0.numel())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # normalisation (input PSD only)
    Xs = []
    for i, (xb, _) in enumerate(loader):
        Xs.append(xb)
        if i >= 4: break
    Xp = torch.cat(Xs, 0).float()
    x_mean, x_std = Xp.mean(0), Xp.std(0).clamp_min(1e-6)

    def nx(x): return (x - x_mean) / x_std

    # tensors for simulator
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    L_t = torch.tensor(leadfield, dtype=torch.float32, device=device)
    gl_t = torch.tensor(group_labels, dtype=torch.long, device=device)

    # build bounds tensors for output mapping
    G = int(group_labels.max()) + 1
    lows_list, highs_list = [], []
    for _ in range(G):
        for k in JR_KEYS:
            lo, hi = bounds[k]
            lows_list.append(lo); highs_list.append(hi)
    # global g as last
    lows_list.append(bounds['g'][0]); highs_list.append(bounds['g'][1])
    lows = torch.tensor(lows_list, dtype=torch.float32, device=device)
    highs = torch.tensor(highs_list, dtype=torch.float32, device=device)

    model = JRParamRegressor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        total = 0.0
        for xb, _ in loader:
            xb = nx(xb.float()).to(device)              # target PSD (normalised)
            # predict params
            y_pred = model(xb)
            # map to bounds
            theta_bounded = _sigmoid_to_bounds(y_pred, lows, highs)
            # run simulator to get PSD from predicted params
            x_hat, fband = torch_simulate_jr_network_psd(
                theta_bounded,
                W=W_t,
                group_labels=gl_t,
                leadfield=L_t,
                base_params=base_params,
                duration=duration,
                dt=dt,
                sfreq_out=sfreq_out,
                fmin=fmin, fmax=fmax,
                nperseg=512, noverlap_frac=0.5,
                noise_strength=0.0,
            )
            # normalise hat with same stats
            x_hat = (x_hat - x_mean.to(device)) / x_std.to(device)
            # frequency weights (alpha ×2)
            with torch.no_grad():
                w = torch.ones_like(fband)
                w = torch.where((fband >= 8) & (fband <= 13), 2.0 * w, w)
                # repeat per channel
                n_ch = x_hat.shape[1] // fband.numel()
                w_rep = w.repeat(n_ch)
            # apply weights channel‑wise
            loss = F.mse_loss(x_hat * w_rep, xb * w_rep)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"[Sim-loss] epoch {ep+1}/{epochs} mse={total/len(dataset):.4f}")

    stats = {"x_mean": x_mean.cpu(), "x_std": x_std.cpu(), "G": int(group_labels.max())+1}
    return model, stats


def infer_params_with_regressor(
    model: JRParamRegressor,
    stats: dict,
    real_eeg: mne.io.Raw,
    fmin: float = 1., fmax: float = 45.,
) -> tuple[dict[int, JRParams], float]:
    """Estimate per-group JR parameters + global g from a real EEG Raw.
    Returns (per_group_params, g_val).
    """
    model.eval()
    X = real_eeg.get_data()  # (n_ch, n_times)
    sfreq = float(real_eeg.info["sfreq"])
    X = mne.filter.filter_data(X, sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose=False)
    feats, _ = compute_psd_features(X, sfreq, fmin=fmin, fmax=fmax)
    x = torch.from_numpy(feats).float()
    x = (x - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        y_hat_norm = model(x.unsqueeze(0)).squeeze(0)
    # denormalise back to parameter space
    y_hat = y_hat_norm * stats["y_std"] + stats["y_mean"]
    per_group, g_val = _vector_to_params(y_hat.cpu().numpy(), stats["G"], JRParams())
    return per_group, g_val

def train_amortized_param_loss(
    n_sims: int,
    duration: float,
    dt: float,
    W: np.ndarray,
    leadfield: np.ndarray,
    group_labels: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    base_params: JRParams | None,
    sfreq_out: float,
    *,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,
    hidden: int = 512,
    depth: int = 3,
    fmin: float = 1.,
    fmax: float = 45.,
    device: str | None = None,
):
    """Train regressor x(PSD) → θ using supervised loss on **parameters**.
    Loss = SmoothL1( zscore(ŷ), zscore(y) ) with running stats from a probe set.
    """
    if base_params is None:
        base_params = JRParams()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticJRDataset(
        n_samples=n_sims, duration=duration, dt=dt, W=W, leadfield=leadfield,
        group_labels=group_labels, bounds=bounds, base_params=base_params,
        sfreq_out=sfreq_out, fmin=fmin, fmax=fmax,
    )

    # Determine dims and compute normalization stats from a few batches
    loader_probe = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    xs, ys = [], []
    for i, (xb, yb) in enumerate(loader_probe):
        xs.append(xb.float()); ys.append(yb.float())
        if i >= 4:  # ~5 batches is enough for stable stats
            break
    Xp = torch.cat(xs, dim=0)
    Yp = torch.cat(ys, dim=0)
    x_mean, x_std = Xp.mean(0), Xp.std(0).clamp_min(1e-6)
    y_mean, y_std = Yp.mean(0), Yp.std(0).clamp_min(1e-6)

    in_dim = int(Xp.shape[1]); out_dim = int(Yp.shape[1])

    def nx(x): return (x - x_mean) / x_std
    def ny(y): return (y - y_mean) / y_std

    model = JRParamRegressor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb = nx(xb.float()).to(device)
            yb = ny(yb.float()).to(device)
            y_pred = model(xb)
            loss = F.smooth_l1_loss(y_pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"[Param-loss] epoch {ep+1}/{epochs} loss={total/len(dataset):.4f}")

    stats = {"x_mean": x_mean.cpu(), "x_std": x_std.cpu(), "y_mean": y_mean.cpu(), "y_std": y_std.cpu(), "G": int(group_labels.max())+1}
    return model, stats

###############################################################################
# 5. MAIN DEMO
###############################################################################
    
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    sim_t=60.0 # seconds to simulate
    dt = 1/512.0  # simulation time step (Hz)
    
    real_raw = mne.io.read_raw_fif("/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/eval/normal/aaaaacad_s003_t000_eeg.fif", preload=True, verbose=False)
    real_chs = real_raw.info["ch_names"]     
    
    sfreq = float(real_raw.info["sfreq"])

    # keep only the ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'] channels
    real_chs = [ch for ch in real_chs if ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']]
    real_raw.pick_channels(real_chs)  # keep only the channels we want
    # Choose parcellation & connectivity
    atlas_img, labels, centroids = load_parcellation(n_parcels=100)

    sanity_check_centroids(centroids, atlas_img)
    # ------------------------------------------------------------------
    # Optionally drop out‑of‑mask parcels
    # ------------------------------------------------------------------
    if PRUNE_OUT_OF_MASK:
        atlas_data = atlas_img.get_fdata()
        aff        = atlas_img.affine
        ijk_coords = np.linalg.inv(aff) @ np.c_[centroids, np.ones(len(centroids))].T
        mask_vals  = map_coordinates(atlas_data, ijk_coords[:3], order=1)
        valid_mask = mask_vals > 0

        if not np.all(valid_mask):
            bad_idx = np.where(~valid_mask)[0]
            print(f"Removing {len(bad_idx)} out‑of‑mask parcels: {bad_idx}")
            centroids = centroids[valid_mask]
            valid_mask_bool = valid_mask  # keep for pruning W later
        else:
            valid_mask_bool = None
    else:
        print("PRUNE_OUT_OF_MASK=False → keeping all parcels, even if outside atlas mask.")
        valid_mask_bool = None

    # update n after possible pruning
    n = centroids.shape[0]

    np.random.seed(0)

    W, labels_ctx, _, _ = load_sc(parcellation='schaefer_100')  # shape (100, 100)
    if valid_mask_bool is not None:
        W = W[np.ix_(valid_mask_bool, valid_mask_bool)]

    # --- normalise & clean ---------------------------------------
    
    # W = (W + W.T) / 2               # enforce symmetry
    # np.fill_diagonal(W, 0)
    # W /= W.max()      

    # ---------------------------------------------------------------------

    # Forward model to EEG sensors
    leadfield, sim_info = build_forward_matrix(
        centroids,
        montage_name="standard_1020",             # keep the same montage
        pick_ch_names=real_chs,                    # <-- new kwarg
        sfreq=1.0/dt                               # <-- new kwarg
    )

    raw_real = real_raw.copy().pick_types(eeg=True)
    
    # Visual check: sensors vs. centroids
    #plot_sensors_vs_centroids(centroids, sim_info)
    
    
    # ---------------------------------------------------------------------
    group_labels = pg.hierarchical_grouping(centroids, W, n_groups=2, alpha=0.5)
    #group_labels = np.repeat(np.arange(4), n // 4)  # 4 groups, each with n/4 parcels
    print("Group sizes:", np.bincount(group_labels))

    # ---------------------------------------------------------------
    # Amortized regression (no CMA‑ES)
    # ---------------------------------------------------------------
    bounds = {
        "A": (0.5, 5.0),
        "B": (10.0, 30.0),
        "a": (60.0, 140.0),
        "b": (30.0, 70.0),
        "C": (80.0, 200.0),
        "p_mean": (90.0, 140.0),
        "g": (0.05, 0.4),
    }

    print("\n==== Amortized regression mode ====")
    model, stats = train_amortized_param_loss(
        n_sims=50,
        duration=sim_t,
        dt=dt,
        W=W,
        leadfield=leadfield,
        group_labels=group_labels,
        bounds=bounds,
        base_params=JRParams(),
        sfreq_out=sfreq,
        batch_size=32,
        epochs=15,
        lr=1e-3,
    )
    best_params, g_val = infer_params_with_regressor(model, stats, real_raw, fmin=1., fmax=45.)
    print("Inferred global coupling g:", g_val)

    # Plot comparison of simulated and real EEG using the fitted parameters
    t, v = simulate_jr_network(
        duration=sim_t, dt=dt, W=W,
        group_labels=group_labels,
        per_group_params=best_params,
        coupling_gain=g_val,
    )
    eeg_sim = parcel_to_sensor(v, leadfield)
    raw_sim = mne.io.RawArray(eeg_sim, sim_info, first_samp=0, copy="auto")
    raw_sim = raw_sim.copy().resample(sfreq, npad="auto")
    #raw_sim = raw_sim.copy().filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)
    raw_sim.plot(title="Simulated EEG (fitted)", duration=sim_t, n_channels=20)
    raw_real = real_raw.copy().pick_types(eeg=True)
    raw_real.plot(title="Real EEG", duration=sim_t, n_channels=20)
    plt.savefig("simulated_vs_real_eeg.png")

    # --- Normalised PSD comparison (shape only) ---
    from scipy.signal import welch as _welch
    # get sensor arrays
    sim_eeg = raw_sim.get_data()
    real_eeg = raw_real.get_data()
    # compute PSDs
    f_sim, P_sim = _welch(sim_eeg, fs=sfreq, axis=1, nperseg=512)
    f_real, P_real = _welch(real_eeg, fs=sfreq, axis=1, nperseg=512)
    # restrict to common band 1–40 Hz
    band_sim = (f_sim >= 1) & (f_sim <= 40)
    band_real = (f_real >= 1) & (f_real <= 40)
    f = f_sim[band_sim]
    P_sim = P_sim[:, band_sim]
    P_real = P_real[:, band_real]
    # per‑channel L1 normalisation (shape only)
    P_sim = P_sim / (P_sim.sum(axis=1, keepdims=True) + 1e-20)
    P_real = P_real / (P_real.sum(axis=1, keepdims=True) + 1e-20)
    # plot channel‑averaged curves
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, P_sim.mean(axis=0), label='Simulated (norm)', alpha=0.8)
    plt.semilogy(f, P_real.mean(axis=0), label='Real (norm)', alpha=0.8)
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Normalised PSD (a.u.)')
    plt.title('Simulated vs Real EEG PSD (shape‑normalised)')
    plt.legend(); plt.tight_layout(); 
    plt.savefig("simulated_vs_real_eeg_psd_parcellation.png")
    # ---------------------------------------------------------------------