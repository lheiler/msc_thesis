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
import cma
import nibabel as nib
from nibabel.affines import apply_affine
from collections import defaultdict
import parcel_grouping as pg
import time
from scipy.signal import welch


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
    nperseg = min(2**next_fast_len(min(sim.shape[1], real.shape[1])), 1024)
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

    loss = np.mean((np.log(psd_sim) - np.log(psd_real))**2)
    return loss

def lf_alpha_weighted(sim, real, sfreq):
    f, P_sim = welch(sim, sfreq, axis=1, nperseg=1024)
    _, P_real = welch(real, sfreq, axis=1, nperseg=1024)
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

def psd_log_mse(sim, real, sfreq, nperseg=1024, fmin=1, fmax=45):
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
    
    # optional per-channel normalisation (makes scale differences irrelevant)
    P_sim /= P_sim.sum(axis=1, keepdims=True)
    P_real /= P_real.sum(axis=1, keepdims=True)
    

    # log-space MSE (matches what you eye-ball in semilogy)
    loss = np.mean((np.log10(P_sim) - np.log10(P_real))**2)
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
    f_sim, P_sim = welch(sim, sfreq, axis=1, nperseg=1024)
    _,    P_real = welch(real, sfreq, axis=1, nperseg=1024)
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


def fit_to_eeg(
    real_eeg: mne.io.Raw,
    leadfield: np.ndarray,
    W: np.ndarray,
    centroids: np.ndarray,
    *,
    group_labels: np.ndarray | None = None,
    params_init: JRParams | None = None,
    bounds: dict[str, Tuple[float, float]] | None = None,
    duration: float | None = None,
    dt: float = 1e-3,
):
    """Fit JR parameters to empirical EEG by minimising mean squared error of sensor signals.

    *This is a toy implementation.* Replace with frequency-domain or connectivity-based
    metrics for more robust fitting, and use a smarter optimiser (e.g. `scipy.optimize.differential_evolution`).
    """
    if duration is None:
        duration = real_eeg.times[-1]
    data = real_eeg.get_data()
    sfreq = real_eeg.info["sfreq"]

    if group_labels is None:
        group_labels = np.zeros(W.shape[0], dtype=int)
    G = int(group_labels.max()) + 1

    if params_init is None:
        params_init = JRParams()
    if bounds is None:
        bounds = {
            "A": (0.5, 5.0),      # pyramidal‑to‑excitatory gain
            "B": (10.0, 30.0),    # pyramidal‑to‑inhibitory gain
            "a": (60.0, 140.0),   # excitatory inverse time‑constant
            "b": (30.0, 70.0),    # inhibitory inverse time‑constant
            "C": (80.0, 200.0),   # average synaptic contacts
            "p_mean": (90.0, 140.0),  # **NEW** external drive (was fixed)
            "g": (0.05, 0.4),  # global coupling gain
        }

    keys  = list(bounds.keys())
    jr_keys = [k for k in keys if k != "g"]
    p0    = np.hstack([[getattr(params_init, k) for k in jr_keys] for _ in range(G)] + [DEFAULT_COUPLING_GAIN])
    lower = np.hstack([[bounds[k][0]          for k in jr_keys] for _ in range(G)] + [bounds["g"][0]])
    upper = np.hstack([[bounds[k][1]          for k in jr_keys] for _ in range(G)] + [bounds["g"][1]])

    recorded_params = []
    recorded_losses = []


    def _loss(x: np.ndarray) -> float:
        # last element is coupling gain
        g_val = x[-1]
        x_groups = x[:-1].reshape(G, len(jr_keys))
        per_group_params = {}
        for g in range(G):
            updated = params_init.__dict__.copy()
            for i, k in enumerate(jr_keys):
                updated[k] = x_groups[g, i]
            per_group_params[g] = JRParams(**updated)

        t, parcel_ts = simulate_jr_network(
            duration, dt, W,
            params=params_init,
            group_labels=group_labels,
            per_group_params=per_group_params,
            coupling_gain=g_val,
        )
        sim_eeg = parcel_to_sensor(parcel_ts, leadfield)
        # resample to match empirical sfreq
        sim_sfreq = 1.0 / dt
        sim_eeg = mne.filter.resample(sim_eeg, down=sim_sfreq / sfreq, axis=1)
        #filter
        sim_eeg = mne.filter.filter_data(sim_eeg, sfreq=sfreq, l_freq=1.0, h_freq=40.0, verbose=False)
        
        
        target = data[:, : sim_eeg.shape[1]]

        
        #loss = psd_log_mse(sim_eeg, target, sfreq, nperseg=1024, fmin=1, fmax=45)
        loss = correlation_lf(sim_eeg, target)

        #print("Current loss:", loss, "params:", per_group_params)
        print("Current loss:", loss)
        print()
        recorded_params.append(per_group_params)
        recorded_losses.append(loss)
        return loss


    x0     = p0                            # same starting point you used before
    sigma  = 0.25 * (upper - lower).mean() # rough initial step size
    bounds = [lower.tolist(), upper.tolist()]                # CMA needs [lo, hi] arrays

    
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma,
        {'bounds': bounds, 'popsize': 16, 'maxfevals': 3000}
    )

    es.optimize(_loss, verb_disp=1, iterations=100)  # assuming _loss(x) → float

    best_x = es.result.xbest
    #get best parameters

    best_idx   = np.argmin(recorded_losses)
    best_params = recorded_params[best_idx]
    print("Best parameters:", best_params)
    print("Best loss:", recorded_losses[best_idx])
    return best_params

###############################################################################
# 5. MAIN DEMO
###############################################################################
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    sim_t=60.0 # seconds to simulate
    dt = 1/512.0  # simulation time step (Hz)
    
    real_raw = mne.io.read_raw_fif("/Users/lorenzheiler/small_dataset_clean/eval/normal/aaaaaayx_s002_t000_eeg.fif", preload=True, verbose=False)
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


    #----------------------------- plot example params ----------------------------
    # Simulate the Jansen-Rit network with default parameters
    #plot model with params: {0: JRParams(A=1.001194228752593, B=16.33640012216592, a=98.37898919090514, b=54.9541750866581, C=145.61525284899693, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56), 1: JRParams(A=1.8585908116569998, B=13.285864283948904, a=95.01679984230542, b=69.6981106994753, C=148.98501036918927, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56), 2: JRParams(A=2.188388034575233, B=27.227306208885814, a=109.15730333872663, b=67.0011888904187, C=98.06053226331592, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56), 3: JRParams(A=0.6543828398635331, B=29.999732662269572, a=72.53224872720281, b=38.29266752963216, C=138.10187884134717, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56)}
    
    # params = {
    #     0: JRParams(A=1.001, B=16.336, a=98.379, b=54.954, C=145.615, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56),
    #     1: JRParams(A=1.859, B=13.286, a=95.017, b=69.698, C=148.985, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56),
    #     2: JRParams(A=2.188, B=27.227, a=109.157, b=67.001, C=98.061, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56),
    #     3: JRParams(A=0.654, B=30.000, a=72.532, b=38.293, C=138.102, p_mean=120.0, p_std=30.0, v0=6.0, e0=2.5, r=0.56)
    # }   
    # # Simulate the Jansen-Rit network with the given parameters
    # t, v = simulate_jr_network(
    #     duration=sim_t, dt=dt, W=W,
    #     params=JRParams(),
    #     group_labels=group_labels,
    #     per_group_params=params
    # )
    # Convert parcel activity to EEG sensors
    # eeg_sim = parcel_to_sensor(v, leadfield)    
    # print(real_raw.info["sfreq"])
    # eeg_sim = mne.filter.resample(eeg_sim, down=(1.0/dt) / sfreq, axis=1)
    # #plot PSD comparison of simulated and real EEG
    # raw_sim = mne.io.RawArray(eeg_sim, sim_info, first_samp=0, copy="auto")
    # sim_psd = raw_sim.compute_psd(method='welch', fmin=1, fmax=40, n_fft=128)
    # real_psd = real_raw.compute_psd(method='welch', fmin=1, fmax=40, n_fft=128)
    # sim_psd.plot(show=False)
    # real_psd.plot(show=False)
    # plt.show()
    
    # ---------------------------------------------------------------------




    best_params = fit_to_eeg(
        real_raw, leadfield, W, centroids,
        group_labels=group_labels,
        duration=sim_t, dt=dt
    )
    print(best_params)

    #plot comparison of simulated and real EEG
    t, v = simulate_jr_network(
        duration=sim_t, dt=dt, W=W,
        group_labels=group_labels,
        per_group_params=best_params
    )
    eeg_sim = parcel_to_sensor(v, leadfield)
    raw_sim = mne.io.RawArray(eeg_sim, sim_info, first_samp=0, copy="auto")
    raw_sim = raw_sim.copy().resample(sfreq, npad="auto")
    #raw_sim = raw_sim.copy().filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)
    raw_sim.plot(title="Simulated EEG (fitted)", duration=sim_t, n_channels=20)
    raw_real = real_raw.copy().pick_types(eeg=True)
    raw_real.plot(title="Real EEG", duration=sim_t, n_channels=20)
    plt.show()

    #plot comparison of simulated and real EEG PSD
    sim_psd = raw_sim.compute_psd(method='welch', fmin=1, fmax=40, n_fft=128)
    real_psd = raw_real.compute_psd(method='welch', fmin=1, fmax=40, n_fft=128)
    sim_psd.plot(show=False)
    real_psd.plot(show=False)
    plt.suptitle("Simulated vs Real EEG PSD")
    plt.show() 
    # ---------------------------------------------------------------------