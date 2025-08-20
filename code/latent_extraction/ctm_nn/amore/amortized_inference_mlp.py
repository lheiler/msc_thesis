import argparse
import pathlib
import json
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import mne

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from utils.util import (
    STANDARD_EEG_CHANNELS,
    PSD_CALCULATION_PARAMS,
    clean_raw_eeg,
    compute_psd_from_raw,
    normalize_psd_torch,
    normalize_psd,
    select_device,
    plot_psd_comparison,
)
# -----------------------------------------------------------------------------
# CTM analytic constants (from nn_ctm_parameters.py)
# -----------------------------------------------------------------------------
Lx = Ly = 0.5       # metres
k0 = 10.0           # m^-1
gamma_e = 116.0     # s^-1
r_e = 0.086         # metres (86 mm)
M = 10              # spatial truncation

# Pre-compute the spatial grid (constant, CPU). We move it to device on demand.
_m = torch.arange(-M, M + 1, dtype=torch.float32)
_kx = 2 * torch.pi * _m[:, None] / Lx  # (21,1)
_ky = 2 * torch.pi * _m[None, :] / Ly  # (1,21)
_k2 = _kx ** 2 + _ky ** 2              # (21,21)
_Fk = torch.exp(-_k2 / k0 ** 2)        # (21,21)
_Delta_k = (2 * torch.pi / Lx) * (2 * torch.pi / Ly)


def compute_ctm_psd(params: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Compute CTM analytic power spectrum.

    Parameters
    ----------
    params : (N, 8) tensor of *physical* parameters in the same order as bounds.
    freqs  : (F,) tensor of frequencies in Hertz (must be 1-D).

    Returns
    -------
    psd : (N, F) tensor (real, float32)
    """
    device = params.device
    dtype = params.dtype

    # Unpack parameters (shape (N,1,1,1) for seamless broadcasting with omega (1,F,1,1))
    G_ee, G_ei, G_ese, G_esre, G_srs, alpha, beta, t0 = [
        p.view(-1, 1, 1, 1) for p in params.split(1, dim=1)
    ]

    # Pre-computed spatial grid moved to the same device / dtype
    k2 = _k2.to(device=device, dtype=dtype)           # (21,21)
    Fk = _Fk.to(device=device, dtype=dtype)

    omega = 2 * torch.pi * freqs.to(device=device, dtype=dtype)  # (F,)
    omega = omega.view(1, -1, 1, 1)                             # (1,F,1,1)

    # Broadcast helpers
    N = params.shape[0]

    # L(omega)
    Lw = 1.0 / ((1.0 - 1j * omega / alpha) * (1.0 - 1j * omega / beta))  # (N,F,1,1) complex

    # q^2 r_e^2 term (real)
    num = (1.0 - 1j * omega / gamma_e) ** 2 - 1.0
    den = 1.0 - G_ei * Lw
    bracket = (
        Lw * G_ee
        + (Lw ** 2 * G_ese + Lw ** 3 * G_esre) * torch.exp(1j * omega * t0) / (1.0 - Lw ** 2 * G_srs)
    )
    q2 = (num - bracket / den).real  # (N,F,1,1)

    # Denominator for phi
    denom = (1.0 - G_srs * Lw ** 2) * (1.0 - G_ei * Lw) * (k2 * r_e ** 2 + q2 * r_e ** 2)

    phi = G_ese * torch.exp(1j * omega * t0 / 2.0) / denom  # (N,F,21,21)

    P = torch.sum(torch.abs(phi) ** 2 * Fk, dim=(-2, -1))  # (N,F)
    return (P * _Delta_k).real.float()

###############################################################################
#                         MODEL & SIMULATOR                                   #
###############################################################################

def _psd_input_dim() -> int:
    n_fft = int(PSD_CALCULATION_PARAMS.get("n_fft", 256))
    return n_fft // 2 + 1

# Frequency grid for the CTM forward model to match utils Welch bins exactly
SFREQ = float(PSD_CALCULATION_PARAMS.get("sfreq", 128.0))
N_FREQS = _psd_input_dim()
FREQS = np.linspace(0.0, SFREQ / 2.0, N_FREQS, dtype=np.float32)

PARAM_NAMES = [
    "G_ee",   # Excitatory-to-excitatory gain
    "G_ei",   # Excitatory-to-inhibitory gain
    "G_ese",  # Excitatory specific gain
    "G_esre", # Excitatory specific recurrent gain
    "G_srs",  # Specific-to-reticular gain
    "alpha",  # Excitatory dendritic rate (s⁻¹)
    "beta",   # Inhibitory dendritic rate (s⁻¹)
    "t0",     # Cortico-thalamic delay (s)
]
PARAM_DIM = len(PARAM_NAMES)

# Physiologically plausible ranges (uniform priors)
PRIOR_LOW  = np.array([0.0, -30.0, 0.0, -10.0, -1.0, 10.0, 100.0, 0.01])
PRIOR_HIGH = np.array([30.0,   0.0, 10.0,   0.0,  0.0, 100.0, 400.0, 0.20])



def _ctm_transfer_function(theta: np.ndarray, freqs: np.ndarray = FREQS) -> np.ndarray:
    """Compute analytic PSD of the *coupled* Robinson CTM (8 parameters).

    This is a thin wrapper around `compute_ctm_psd` from
    `code.latent_extraction.ctm_nn.nn_ctm_parameters`, returning a NumPy array.
    """
    th_t = torch.as_tensor(theta, dtype=torch.float32).unsqueeze(0)  # (1, 8)
    freqs_t = torch.as_tensor(freqs, dtype=torch.float32)
    with torch.no_grad():
        psd_t = compute_ctm_psd(th_t, freqs_t)  # (1, F)
    return psd_t.squeeze(0).cpu().numpy()


def _is_stable(theta: np.ndarray) -> bool:
    """Placeholder stability criterion – currently accepts all samples."""
    return True


###############################################################################
#                       DATA GENERATION                                       #
###############################################################################

def sample_prior(n: int) -> np.ndarray:
    """Draw *n* samples uniformly from the 8-D prior."""
    u = np.random.uniform(size=(n, PARAM_DIM))
    return PRIOR_LOW + u * (PRIOR_HIGH - PRIOR_LOW)


def _normalise(psd):
    """Log10 + per-vector z-score (supports single vector (F,) or batch (N,F)).

    - Torch input: stays on-device, preserves dtype
    - NumPy input: returns NumPy array
    """
    # Torch path
    if isinstance(psd, torch.Tensor):
        return normalize_psd_torch(psd)
    else:
        return normalize_psd(psd)

def generate_dataset(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (theta, psd) pairs until *n* stable samples are collected."""
    
    if os.path.exists("/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/data/generated_dataset.npz"):
        data = np.load("/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/data/generated_dataset.npz")
        thetas = data["thetas"]
        psds = data["psds"]
        return thetas, psds
    
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
    #save the dataset to a file
    st, sp = np.stack(thetas, axis=0), np.stack(psds, axis=0)
    np.savez("/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/data/generated_dataset.npz", thetas=st, psds=sp)
    print("saving dataset to /rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/data/generated_dataset.npz")
    return st, sp

###############################################################################
#                            NEURAL NETWORK                                   #
###############################################################################

class ParameterRegressor(torch.nn.Module):
    """Simple feedforward network that maps a PSD to CTM parameters."""

    def __init__(self, in_dim: int = _psd_input_dim(), hidden_dims=(512, 256), out_dim: int = PARAM_DIM):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev, h))
            layers.append(torch.nn.ReLU())
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)



###############################################################################
#                             TRAINING LOOP                                   #
###############################################################################

def train(
    out_dir: pathlib.Path,
    device: torch.device,
    num_sims=100000,
    epochs: int = 50,
    batch_size: int = 1024,
    test_fraction: float = 0.1,
):
    """Train the parameter regressor **and evaluate on a held-out test split**.

    The generated synthetic dataset is randomly split into a training and a test
    set (default 90 % / 10 %). Training metrics are printed each epoch and the
    final test-set Mean-Squared-Error (MSE) is reported after training.
    """

    print(f"[INFO] Generating {num_sims} simulations …")
    theta_np, x_np = generate_dataset(num_sims)
    print(theta_np.shape)
    print(x_np.shape)
    
    # Debug: Check for NaN or inf values
    print(f"[DEBUG] Input PSD stats - min: {np.min(x_np):.2e}, max: {np.max(x_np):.2e}, mean: {np.mean(x_np):.2e}")
    print(f"[DEBUG] Input PSD has NaN: {np.any(np.isnan(x_np))}, has inf: {np.any(np.isinf(x_np))}")

    # Shuffle & split indices
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(num_sims)
    n_test = int(num_sims * test_fraction)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    x_train = torch.as_tensor(x_np[train_idx], dtype=torch.float32)
    t_train = torch.as_tensor(theta_np[train_idx], dtype=torch.float32)
    x_test = torch.as_tensor(x_np[test_idx], dtype=torch.float32)
    t_test = torch.as_tensor(theta_np[test_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, t_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(x_test, t_test), batch_size=batch_size, shuffle=False)

    # Model & optimiser
    model = ParameterRegressor().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use custom PSD loss instead of MSE on parameters
    freqs_tensor = torch.as_tensor(FREQS, dtype=torch.float32, device=device)

    def psd_loss(params: Tensor, target_psd: Tensor) -> Tensor:
        """Reconstruction loss on the log‑normalised power spectral density (PSD).

        Converts the predicted CTM parameters to a PSD via the analytic
        forward model, applies the same log‑zscore normalisation as the
        training data, and computes an MSE with the target PSD.
        """
        # Forward model: parameters → PSD
        pred_psd = compute_ctm_psd(params, freqs_tensor)          # (B,F)

        # Restrict comparison to configured frequency band
        fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
        fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", float(SFREQ) / 2.0))
        mask = (freqs_tensor >= fmin) & (freqs_tensor <= fmax)
        pred_psd = pred_psd[:, mask]
        target_psd = target_psd[:, mask]

        # Log‑transform & z‑score to match `_normalise`
        pred_psd = _normalise(pred_psd)
        target_psd = _normalise(target_psd)
        return torch.nn.functional.mse_loss(pred_psd, target_psd)

    loss_fn = psd_loss

    patience=5
    best_loss=float('inf')
    epochs_no_improve=0

    print("[INFO] Training feedforward network with PSD reconstruction loss …")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, tb in train_loader:
            xb = xb.to(device)
            tb = tb.to(device)
            optim.zero_grad()
            pred = model(xb)
            # Use PSD reconstruction loss instead of parameter MSE
            loss = loss_fn(pred, xb)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss detected: {loss.item()}")
                print(f"[DEBUG] pred range: [{pred.min().item():.2e}, {pred.max().item():.2e}]")
                print(f"[DEBUG] xb range: [{xb.min().item():.2e}, {xb.max().item():.2e}]")
                continue
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()
            train_losses.append(loss.item())

        
        
        # Evaluate on test split each epoch
        model.eval()
        with torch.no_grad():
            if epoch % 1 == 0 or epoch == 1 or epoch == epochs:
                test_losses = []
                for xb, tb in test_loader:
                    xb = xb.to(device)
                    tb = tb.to(device)
                    pred = model(xb)
                    # Use PSD reconstruction loss for evaluation too
                    test_loss = loss_fn(pred, xb)
                    if not (torch.isnan(test_loss) or torch.isinf(test_loss)):
                        test_losses.append(test_loss.item())
                    
            if epoch % 5 == 0:
                with torch.no_grad():
                    pred_psd_full = compute_ctm_psd(pred, torch.as_tensor(FREQS, dtype=torch.float32, device=device))
                    # mask for plotting consistency
                    fmin = float(PSD_CALCULATION_PARAMS.get("min_freq", 0.0))
                    fmax = float(PSD_CALCULATION_PARAMS.get("max_freq", float(SFREQ) / 2.0))
                    mask_np = (FREQS >= fmin) & (FREQS <= fmax)
                    pred_psd = _normalise(pred_psd_full[:, mask_np])
                    xb_masked = _normalise(xb[:, mask_np])
                plot_psd_comparison(xb_masked, pred_psd, FREQS[mask_np], f"/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/results/psd_comparison_{epoch}.png")
            
        print(f"Epoch {epoch:>3d}/{epochs} – train PSD loss: {np.mean(train_losses):.4f} – test PSD loss: {np.mean(test_losses):.4f}")
        
        # Early stopping
        if np.mean(train_losses) < best_loss:
            best_loss = np.mean(train_losses)
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "freqs": FREQS, "param_names": PARAM_NAMES}, "/rds/general/user/lrh24/home/thesis/code/latent_extraction/ctm_nn/amore/models/regressor.pt")
            print("saving model")
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break
        
    final_test_loss = float(np.mean(test_losses)) if test_losses else float('nan')
    print(f"[OK] Training complete – final test PSD loss: {final_test_loss:.4e}")

###############################################################################
#                           INFERENCE                                        #
###############################################################################

# ----------------------------------------------------------------------------
# TUH EEG PSD extraction utility (unchanged)                                   
# ----------------------------------------------------------------------------

def extract_psds_from_tuh(
    root_dir: pathlib.Path,
    preload_path: pathlib.Path = pathlib.Path("data/preloaded_psds.npy"),
    reset: bool = False,
) -> np.ndarray:
    """Extract normalised PSDs from TUH EEG dataset. Each channel PSD is one sample.

    Uses shared preprocessing and Welch parameters via utils.
    """
    fif_files = list(root_dir.rglob("*.fif"))
    print(f"[INFO] Found {len(fif_files)} files in {root_dir}")

    if preload_path.exists() and not reset:
        all_psds = np.load(preload_path)
        if int(all_psds.shape[0] / 19) == len(fif_files):
            return all_psds
        else:
            print(
                f"[INFO] Preloaded PSDs has {all_psds.shape[0]} channels, but {len(fif_files)} files. Recomputing."
            )
    else:
        print(f"[INFO] Preloaded PSDs file does not exist. Recomputing.")

    all_psds: list[np.ndarray] = []

    for fif_path in tqdm(fif_files):
        try:
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
            # Standardise channels and filtering
            raw = clean_raw_eeg(raw)
            # Compute per-channel Welch PSD with shared params and normalisation
            psd_cf = compute_psd_from_raw(raw, calculate_average=False, normalize=True)  # (C,F)
            all_psds.append(psd_cf)
        except Exception as e:
            print(f"[WARN] Skipping {fif_path.name}: {e}")

    if not all_psds:
        raise RuntimeError("No PSDs extracted from TUH directory.")

    stacked = np.concatenate(all_psds, axis=0)  # (N_files*C, F)
    print(stacked.shape)
    np.save(preload_path, stacked)
    return stacked




def infer(model_path: pathlib.Path, data: np.ndarray) -> np.ndarray:
    def smooth_psd(psd: np.ndarray) -> np.ndarray:
        return np.convolve(psd, np.ones(10)/10, mode='same')
    
    model = ParameterRegressor()
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.to("cuda")
    data = torch.as_tensor(data, dtype=torch.float32).to("cuda")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            emp_input = torch.as_tensor(_normalise(smooth_psd(data[i].cpu().numpy())), dtype=torch.float32).to("cuda")
            pred = model(emp_input)[0].cpu().numpy()
            all_preds.append(pred)
            #plot comparison empirical psd with calculated psd  
            # ctm_psd = _ctm_transfer_function(pred)
            # plt.plot(FREQS, emp_input.cpu().numpy(), label="Empirical")
            # plt.plot(FREQS, ctm_psd, label="Calculated")
            # plt.legend()
            # plt.savefig(f"results/psd_comparison_{i}.png")
            # plt.close()
    return np.array(all_preds)
if __name__ == "__main__":
    train(out_dir="models", device="cuda", epochs=70, num_sims=200000, test_fraction=0.1, batch_size=1024)
    # data = extract_psds_from_tuh(pathlib.Path("/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean"))
    # preds = infer(pathlib.Path("models/regressor.pt"), data)
