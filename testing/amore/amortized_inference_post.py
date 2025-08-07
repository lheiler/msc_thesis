import argparse
import pathlib
import json
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import corner
from tqdm import tqdm
import mne
from scipy.signal import welch

# sbi is optional at import-time because users might only want to generate data
try:
    from sbi import utils as sbi_utils
    from sbi import inference as sbi_inference
except ImportError as e:
    sbi_utils = None  # type: ignore
    sbi_inference = None  # type: ignore


###############################################################################
#                         MODEL & SIMULATOR                                   #
###############################################################################

FREQ_MIN = 1.0  # Hz
FREQ_MAX = 45.0  # Hz
N_FREQS = 300  # grid resolution
FREQS = np.linspace(FREQ_MIN, FREQ_MAX, N_FREQS)

# Parameter names for nicer printing & plotting
PARAM_NAMES = [
    "g_e",  # Excitatory loop gain
    "g_i",  # Inhibitory loop gain
    "tau_e",  # Excitatory synaptic time-constant (s)
    "tau_i",  # Inhibitory synaptic time-constant (s)
    "delay",  # Corticothalamic conduction delay (s)
    "sigma",  # Noise scale (arbitrary)
]
PARAM_DIM = len(PARAM_NAMES)

# Physiologically plausible ranges (uniform priors)
PRIOR_LOW = np.array([8.0, 3.0, 0.002, 0.005, 0.005, 1e-7])
PRIOR_HIGH = np.array([40.0, 25.0, 0.01, 0.05, 0.03, 1e-3])


def _ctm_transfer_function(theta: np.ndarray, freqs: np.ndarray = FREQS) -> np.ndarray:
    """Compute analytic PSD of the Robinson Cortico-Thalamic model (simplified).

    The formulation is a reduced two-loop approximation often used in teaching
    examples. It captures a resonant alpha peak and overall 1/f-like fall-off.

    Parameters
    ----------
    theta
        Parameter vector with ordering given by ``PARAM_NAMES``.
    freqs
        Frequency vector (Hz).

    Returns
    -------
    psd : np.ndarray
        Power spectral density (same shape as ``freqs``).
    """
    g_e, g_i, tau_e, tau_i, delay, sigma = theta.astype(float)

    omega = 2 * np.pi * freqs  # angular frequency

    # Synaptic kernels (second-order low-pass) – Robinson et al. 2004 Eq. 3
    L_e = 1.0 / (1.0 + 1j * omega * tau_e) ** 2
    L_i = 1.0 / (1.0 + 1j * omega * tau_i) ** 2

    # Effective loop gain with conduction delay (phase shift)
    G = g_e * L_e - g_i * L_i * np.exp(-1j * omega * delay)

    # Transfer from white noise to membrane potential – Eq. like 1/(1-G)
    H = 1.0 / (1.0 - G)

    psd = sigma ** 2 * np.abs(H) ** 2
    return psd.real  # PSD is real & positive by construction


def _is_stable(theta: np.ndarray) -> bool:
    """Very crude stability criterion: |G(0)| < 1."""
    g_e, g_i, *_ = theta
    return np.abs(g_e - g_i) < 1.0


def simulate(theta: Tensor) -> Tensor:
    """Torch simulator wrapper for *sbi* (single or batched)."""
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1, P)
    psds = []
    for th in theta.cpu().numpy():
        psd = _ctm_transfer_function(th)
        psd = _normalise(psd)
        psds.append(psd)
    return torch.as_tensor(np.stack(psds, axis=0), dtype=torch.float32, device=theta.device)


###############################################################################
#                       DATA GENERATION                                       #
###############################################################################

def sample_prior(n: int) -> np.ndarray:
    """Draw *n* samples from the prior (uniform / log-uniform for sigma)."""
    u = np.random.uniform(size=(n, PARAM_DIM))
    samples = PRIOR_LOW + u * (PRIOR_HIGH - PRIOR_LOW)
    # Log-uniform for sigma (last param)
    log_sigma_low, log_sigma_high = np.log10(PRIOR_LOW[-1]), np.log10(PRIOR_HIGH[-1])
    samples[:, -1] = 10 ** (
        np.random.uniform(log_sigma_low, log_sigma_high, size=n)
    )
    return samples


def generate_dataset(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (theta, psd) pairs until *n* stable samples are collected."""
    thetas = []
    psds = []
    while len(thetas) < n:
        batch = sample_prior(n)
        for th in batch:
            if not _is_stable(th):
                continue
            psd = _ctm_transfer_function(th)
            psd = _normalise(psd)
            thetas.append(th)
            psds.append(psd)
            if len(thetas) >= n:
                break
    return np.stack(thetas, axis=0), np.stack(psds, axis=0)


###############################################################################
#                         UTILITIES                                           #
###############################################################################

def _normalise(psd: np.ndarray) -> np.ndarray:
    """Log-transform & mean-centre to capture *shape* not absolute magnitude."""
    log_psd = np.log10(psd + 1e-12)
    return log_psd - log_psd.mean()


def _ensure_out_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: pathlib.Path):
    path.write_text(json.dumps(obj, indent=2))


# -----------------------------------------------------------------------------
# Helper to move the internal network of an sbi ``DirectPosterior`` to a device
# -----------------------------------------------------------------------------


def _move_posterior_to_device(posterior, device: torch.device):
    """Move the underlying neural network (and any stray buffers) to *device*.

    In some *sbi* versions the `Standardize` transform keeps `_mean`/`_std` as plain
    tensors that are **not** registered as buffers and therefore remain on CPU when
    calling `.to(device)`. We therefore explicitly look for such attributes and move
    them as well. The procedure is a no-op if they are already on the correct
    device.
    """
    # `DirectPosterior` wraps a density-estimator; depending on the *sbi* version it
    # is accessible via `.net` (>=0.22) or `._log_prob_net` (older).
    net = getattr(posterior, "net", None) or getattr(posterior, "_log_prob_net", None)
    if net is None or not isinstance(device, torch.device):
        return

    # First, move the whole network recursively (registered parameters + buffers).
    net.to(device)

    # Second, handle stray tensors stored as plain attributes.
    def _move_attr_tensor(obj, attr_name: str):
        t = getattr(obj, attr_name, None)
        if isinstance(t, torch.Tensor):
            setattr(obj, attr_name, t.to(device))

    # Convenience: helper to move nn.Module or Tensor-like objects.
    def _move_obj(obj):
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
            # Recursively handle stray tensors inside the module (see below).
            for mod in obj.modules():
                _move_attr_tensor(mod, "_mean"); _move_attr_tensor(mod, "_std"); _move_attr_tensor(mod, "mean"); _move_attr_tensor(mod, "std")
        elif isinstance(obj, torch.Tensor):
            return obj.to(device)
        return obj

    for module in net.modules():
        _move_attr_tensor(module, "_mean")
        _move_attr_tensor(module, "_std")
        _move_attr_tensor(module, "mean")
        _move_attr_tensor(module, "std")

    # Additionally, some *nflows*/*sbi* versions attach auxiliary networks (e.g.,
    # `_embedding_net`) as plain attributes without properly registering them as
    # sub-modules. Such networks (and their parameters/buffers) are therefore *not*
    # moved by the calls above and can remain on CPU, leading to device mismatch
    # errors at inference time. We iterate over all attributes of `net`, look for
    # unregistered `nn.Module` instances, and move them (and their stray tensors)
    # to the requested device.
    def _move_unregistered_submodules(parent: torch.nn.Module):
        """Move any nn.Module attributes of *parent* that are not registered sub-modules."""
        registered = set(parent.modules())
        for attr_name in dir(parent):
            if attr_name.startswith("__"):
                continue  # skip dunder attributes
            submod = getattr(parent, attr_name, None)
            if isinstance(submod, torch.nn.Module) and submod not in registered:
                submod.to(device)
                # Also move potential plain-tensor attributes inside the sub-module
                for module in submod.modules():
                    _move_attr_tensor(module, "_mean")
                    _move_attr_tensor(module, "_std")
                    _move_attr_tensor(module, "mean")
                    _move_attr_tensor(module, "std")

    # First handle net itself.
    _move_unregistered_submodules(net)

    # Additionally, some *sbi* versions keep the neural network under the name
    # `posterior_estimator` instead of `net`/`_log_prob_net`. If present, treat it
    # in the same way.
    estimator = getattr(posterior, "posterior_estimator", None)
    if isinstance(estimator, torch.nn.Module):
        estimator.to(device)
        for module in estimator.modules():
            _move_attr_tensor(module, "_mean")
            _move_attr_tensor(module, "_std")
            _move_attr_tensor(module, "mean")
            _move_attr_tensor(module, "std")
        _move_unregistered_submodules(estimator)

    # --- Fallback: brute-force over all attributes on the posterior itself --------
    for attr_name in dir(posterior):
        if attr_name.startswith("__"):
            continue
        val = getattr(posterior, attr_name, None)
        _move_obj(val)


###############################################################################
#                            TRAINING                                        #
###############################################################################

def train(num_sims: int, out_dir: pathlib.Path, device: torch.device, diagnostics: bool = False):
    if sbi_utils is None or sbi_inference is None:
        raise RuntimeError("sbi is required for training – please `pip install sbi`. ")

    print(f"[INFO] Generating {num_sims} simulations …")
    theta_np, x_np = generate_dataset(num_sims)
    theta = torch.as_tensor(theta_np, dtype=torch.float32, device=device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

    prior = sbi_utils.BoxUniform(
        low=torch.as_tensor(PRIOR_LOW, dtype=torch.float32, device=device),
        high=torch.as_tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
    )

    print("[INFO] Training NPE (normalising-flow posterior)…")
    inference = sbi_inference.SNPE(prior=prior, device=str(device))
    # Disable tensorboard logging; print a brief table instead.
    density_estimator = (
        inference.append_simulations(theta, x)
        .train(show_train_summary=True)
    )
    posterior = inference.build_posterior(density_estimator)
    _move_posterior_to_device(posterior, torch.device("cpu"))  # save on CPU for portability

    _ensure_out_dir(out_dir)
    ckpt_path = out_dir / "posterior.pkl"
    torch.save({"posterior": posterior, "freqs": FREQS, "param_names": PARAM_NAMES}, ckpt_path)
    print(f"[OK] Posterior saved to {ckpt_path}")

    if diagnostics:
        _perform_sbc(posterior, num_sims=1000, out_dir=out_dir)


###############################################################################
#                           INFERENCE                                        #
###############################################################################


def _load_empirical_psd(path: pathlib.Path) -> np.ndarray:
    if path.suffix.lower() in {".csv", ".txt"}:
        arr = pd.read_csv(path, header=None).values.squeeze()
    elif path.suffix.lower() in {".npy"}:
        arr = np.load(path)
    else:
        raise ValueError("Unsupported PSD file format. Use CSV or NPY.")
    assert (
        arr.shape[0] == N_FREQS or (arr.ndim == 2 and arr.shape[1] == N_FREQS)
    ), f"Expected PSD with {N_FREQS} points, got shape {arr.shape}"
    return arr.astype(float)


# -----------------------------------------------------------------------------
# TUH EEG PSD extraction utility
# -----------------------------------------------------------------------------
def extract_psds_from_tuh(root_dir: pathlib.Path, n_freqs: int = N_FREQS, fmin: float = FREQ_MIN, fmax: float = FREQ_MAX) -> np.ndarray:
    """Extract PSDs from TUH EEG dataset directory. Each channel's PSD becomes one sample."""
    TEN_TWENTY_CHANNELS = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T3", "C3", "Cz", "C4", "T4",
        "T5", "P3", "Pz", "P4", "T6",
        "O1", "O2"
    ]
    all_psds = []
    for phase in ["train", "eval"]:
        for label in ["abnormal", "normal"]:
            data_path = root_dir / phase / label
            fif_files = list(data_path.rglob("*.fif"))
            print(f"[INFO] Found {len(fif_files)} files in {data_path}")
            for fif_path in tqdm(fif_files):
                try:
                    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
                    picked_channels = [ch for ch in TEN_TWENTY_CHANNELS if ch in raw.ch_names]
                    if len(picked_channels) < 1:
                        print(f"[WARN] Skipping {fif_path.name}: no 10-20 channels found.")
                        continue
                    raw.pick_channels(picked_channels)
                    raw.set_eeg_reference('average', projection=True)
                    raw.filter(fmin, fmax, fir_design='firwin')
                    freqs, psds = welch(raw.get_data(), fs=raw.info['sfreq'], nperseg=1024)
                    # Ensure psds is 2‑D: (n_channels, n_freqs). Welch returns 1‑D when only one channel is passed.
                    if psds.ndim == 1:
                        psds = psds[np.newaxis, :]
                    # Guard: if only one channel survived the pick, psds will be 1‑D.
                    if psds.ndim == 1:
                        psds = psds[np.newaxis, :]
                    freqs_mask = (freqs >= fmin) & (freqs <= fmax)
                    freqs_selected = freqs[freqs_mask]
                    target_freqs = np.linspace(fmin, fmax, n_freqs)
                    for idx, ch_psd in enumerate(psds):
                        if not isinstance(ch_psd, np.ndarray) or ch_psd.ndim != 1 or len(ch_psd) != len(freqs):
                            print(f"[WARN] Skipping channel {idx} in {fif_path.name}: invalid PSD shape {ch_psd.shape}")
                            continue
                        try:
                            ch_psd_selected = ch_psd[freqs_mask]
                            interp_psd = np.interp(target_freqs, freqs_selected, ch_psd_selected)
                            log_psd = np.log10(interp_psd + 1e-12)
                            log_psd -= log_psd.mean()
                            all_psds.append(log_psd)
                        except Exception as e:
                            print(f"[WARN] Failed to process channel {idx} in {fif_path.name}: {e}")
                except Exception as e:
                    print(f"[WARN] Skipping {fif_path.name}: {e}")
                    
    fin_arr = np.array(all_psds)
    print(fin_arr.shape)
    return fin_arr


def infer(
    psd_path: pathlib.Path,
    out_dir: pathlib.Path,
    n_samples: int,
    device: torch.device,
    diagnostics: bool = False,
):
    ckpt_path = pathlib.Path("models") / "posterior.pkl"  # heuristic default
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Could not find posterior checkpoint at {ckpt_path}. "
            "Please specify the same --out_dir used during training."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    posterior = ckpt["posterior"]
    _move_posterior_to_device(posterior, device)

    if psd_path.is_dir():
        # Cache PSD extraction results to speed up repeated runs
        cache_path = pathlib.Path("models") / "empirical_psds.npy"
        if cache_path.exists():
            print(f"[INFO] Loading cached PSDs from {cache_path}")
            empirical_psd = np.load(cache_path)
        else:
            print(f"[INFO] Extracting PSDs from directory: {psd_path}")
            empirical_psd = extract_psds_from_tuh(psd_path)
            _ensure_out_dir(cache_path.parent)
            np.save(cache_path, empirical_psd)
            print(f"[OK] Saved extracted PSDs to {cache_path}")
    else:
        empirical_psd = _load_empirical_psd(psd_path)

    if empirical_psd.ndim == 1:
        x_all = torch.as_tensor(_normalise(empirical_psd), dtype=torch.float32, device=device).unsqueeze(0)
    else:
        x_all = torch.as_tensor(np.stack([_normalise(p) for p in empirical_psd]), dtype=torch.float32, device=device)

    print(f"[INFO] Drawing {n_samples} posterior samples for {x_all.shape[0]} PSD(s)…")
    samples_all = []
    for x_o in tqdm(x_all):
        samples = posterior.sample((n_samples,), x=x_o).cpu().numpy()
        samples_all.append(samples)
    samples_np = np.stack(samples_all)  # Shape: (N, n_samples, param_dim)

    for i, posterior_samples in enumerate(samples_np):
        means = posterior_samples.mean(axis=0)
        ci_low = np.percentile(posterior_samples, 2.5, axis=0)
        ci_high = np.percentile(posterior_samples, 97.5, axis=0)
        print(f"\n[Sample {i+1}]")
        for name, m, lo, hi in zip(PARAM_NAMES, means, ci_low, ci_high):
            print(f"{name:>6s}: {m:.4g}  [{lo:.4g}, {hi:.4g}]")

    _ensure_out_dir(out_dir)

    # Corner plot
    fig = corner.corner(samples_np[0], labels=PARAM_NAMES, truths=samples_np[0].mean(axis=0))
    fig_path = out_dir / "posterior_corner.png"
    fig.savefig(fig_path, dpi=300)
    print(f"[OK] Corner plot saved to {fig_path}")

    if diagnostics:
        _posterior_predictive_check(samples_np[0], empirical_psd[0], out_dir)


###############################################################################
#                         DIAGNOSTICS                                        #
###############################################################################

def _perform_sbc(posterior, num_sims: int, out_dir: pathlib.Path):
    """Very light-weight Simulation-Based Calibration."""
    print("[INFO] Running SBC (this may take a while)…")
    ranks = np.zeros((num_sims, PARAM_DIM))
    for i in range(num_sims):
        th = sample_prior(1)[0]
        x = _normalise(_ctm_transfer_function(th))
        post_samples = posterior.sample((100,), x=torch.as_tensor(x, dtype=torch.float32))
        ranks[i] = (post_samples.numpy() < th).sum(axis=0)
    # Histogram of ranks (should be uniform)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ranks.flatten(), bins=20, density=True)
    ax.set_title("SBC Rank Histogram")
    fig.savefig(out_dir / "sbc_ranks.png", dpi=300)
    print(f"[OK] SBC plot saved to {out_dir / 'sbc_ranks.png'}")


def _posterior_predictive_check(samples: np.ndarray, psd_emp: np.ndarray, out_dir: pathlib.Path):
    print("[INFO] Performing posterior predictive check …")
    sim_psds = [_normalise(_ctm_transfer_function(th)) for th in samples[:200]]
    sim_psds = np.stack(sim_psds, axis=0)

    mean_sim = sim_psds.mean(axis=0)
    ci_low = np.percentile(sim_psds, 2.5, axis=0)
    ci_high = np.percentile(sim_psds, 97.5, axis=0)

    plt.figure(figsize=(8, 4))
    plt.fill_between(FREQS, ci_low, ci_high, alpha=0.3, label="95% Posterior Pred.")
    plt.plot(FREQS, mean_sim, label="Mean Posterior Pred.")
    plt.plot(FREQS, _normalise(psd_emp), color="k", lw=2, label="Empirical")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalised log-PSD")
    plt.legend()
    plt.tight_layout()
    _ensure_out_dir(out_dir)
    plt.savefig(out_dir / "posterior_predictive.png", dpi=300)
    plt.close()
    print(f"[OK] PPC plot saved to {out_dir / 'posterior_predictive.png'}")


###############################################################################
#                               CLI                                          #
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Amortised Bayesian inference for the Robinson CTM EEG PSD."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------- Train --------------------
    p_train = subparsers.add_parser("train", help="Generate sims & train posterior.")
    p_train.add_argument("--num_sims", type=int, default=200_000)
    p_train.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("models"))
    p_train.add_argument("--diagnostics", action="store_true", help="Run SBC diagnostics.")

    # -------------------- Extract --------------------
    p_ext = subparsers.add_parser("extract", help="Extract PSDs from TUH EEG directory.")
    p_ext.add_argument("--tuh_dir", type=pathlib.Path, required=True, help="Root directory of TUH EEG dataset.")
    p_ext.add_argument("--out_file", type=pathlib.Path, default=pathlib.Path("models/empirical_psds.npy"), help="Path to save extracted PSDs.")

    # -------------------- Infer --------------------
    p_inf = subparsers.add_parser("infer", help="Infer parameters from empirical PSD.")
    p_inf.add_argument("--psd_csv", type=pathlib.Path, required=True, help="Empirical PSD file (CSV or NPY).")
    p_inf.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("results"))
    p_inf.add_argument("--n_samples", type=int, default=10_000)
    p_inf.add_argument("--diagnostics", action="store_true", help="Posterior-predictive check.")

    # Global option: device
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use (default: cuda if available)",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args.num_sims, args.out_dir, torch.device(args.device), diagnostics=args.diagnostics)
    elif args.command == "infer":
        infer(args.psd_csv, args.out_dir, args.n_samples, torch.device(args.device), diagnostics=args.diagnostics)
    elif args.command == "extract":
        psds = extract_psds_from_tuh(args.tuh_dir)
        _ensure_out_dir(args.out_file.parent)
        np.save(args.out_file, psds)
        print(f"[OK] Saved {psds.shape[0]} PSDs to {args.out_file}")
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
