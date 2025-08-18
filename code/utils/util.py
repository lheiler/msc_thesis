import json
import numpy as np
import torch
import mne
import matplotlib.pyplot as plt
# Standard 19-channel EEG montage used across methods
STANDARD_EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Cz', 'Pz', 'Fz'
]

PSD_CALCULATION_PARAMS = {
    "n_fft": 256,
    "n_overlap": 128,
    "n_per_seg": 256,
    "min_freq": 3.0,
    "max_freq": 45.0,
    "segment_length": 60.0,
    "sfreq": 128.0,
    "freqs": np.linspace(3, 45, 256//2 + 1),
}


def clean_raw_eeg(raw, segment_length=60.0):
    """Drop aux channels (A1/A2) and pick the standard 19 EEG channels.
    Mirrors the prior inline cleaning logic used in extractors.
    """
    # Ensure data is loaded for operations like filtering/cropping
    try:
        raw.load_data(verbose=False)
    except Exception:
        # If already loaded or load not supported, continue
        pass
    if 'A1' in raw.ch_names:
        raw.drop_channels(['A1'])
    if 'A2' in raw.ch_names:
        raw.drop_channels(['A2'])
    raw.pick(STANDARD_EEG_CHANNELS)
    raw.crop(tmin=0.0, tmax=segment_length-1.0/raw.info['sfreq'])
    raw.filter(PSD_CALCULATION_PARAMS["min_freq"], PSD_CALCULATION_PARAMS["max_freq"], fir_design="firwin", verbose=False)
    return raw


def select_device():
    """Select CUDA → MPS → CPU, matching existing behavior."""
    return torch.device(
        'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    )


def ensure_float32_tensor(x):
    """Convert array-like to torch.float32 tensor."""
    return torch.as_tensor(x, dtype=torch.float32)


def make_latent_record(latent_feature, gender, age, abnormal):
    """Serialize a latent tuple matching JSONL output format used downstream."""
    vec = latent_feature.tolist() if hasattr(latent_feature, 'tolist') else latent_feature
    g = int(gender.item()) if hasattr(gender, 'item') else int(gender)
    a = int(age.item()) if hasattr(age, 'item') else int(age)
    ab = int(abnormal.item()) if hasattr(abnormal, 'item') else int(abnormal)
    return (vec, g, a, ab)


def truncate_file(path):
    """Create or truncate a file."""
    with open(path, 'w') as f:
        pass


def append_jsonl(path, record_tuple):
    """Append a JSON-serialized record to a JSONL file."""
    with open(path, 'a') as f:
        f.write(json.dumps(record_tuple) + '\n')
        
def normalize_psd(psd: np.ndarray):
    #make it work for both 1D and 2D arrays
    if psd.ndim == 1:
        log_psd = np.log10(psd + 1e-12)
        return (log_psd - log_psd.mean()) / log_psd.std()
    else:
        log_psd = np.log10(psd + 1e-12)
        return (log_psd - log_psd.mean(axis=1, keepdims=True)) / log_psd.std(axis=1, keepdims=True)

def normalize_psd_torch(psd: torch.Tensor):
    #make it work for both 1D and 2D tensors
    if psd.ndim == 1:
        log_psd = torch.log10(psd + 1e-12)
        return (log_psd - log_psd.mean()) / log_psd.std()
    else:
        log_psd = torch.log10(psd + 1e-12)
        return (log_psd - log_psd.mean(dim=1, keepdim=True)) / log_psd.std(dim=1, keepdim=True)



def compute_psd_from_raw(
    raw,
    *,
    n_fft: int = PSD_CALCULATION_PARAMS["n_fft"],
    n_overlap: int = PSD_CALCULATION_PARAMS["n_overlap"],
    n_per_seg: int = PSD_CALCULATION_PARAMS["n_per_seg"],
    calculate_average: bool = False,
    normalize: bool = True,
    return_freqs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD from Raw with shared defaults.

    Args:
        raw: MNE Raw, assumed already channel-cleaned and bandpassed.
        n_fft, n_overlap, n_per_seg: Welch parameters.
        calculate_average: If True, returns a 1D (F,) average across channels. Else (C, F).
        normalize: If True, apply log10 + per-vector z-score. For (C, F) input, normalize per-channel.
        return_freqs: If True, also return the frequency bins as a second output.

    Returns:
        psd: np.ndarray of shape (F,) if averaged else (C, F). If return_freqs, also returns freqs (F,).
    """
    sfreq = float(raw.info.get('sfreq', 128.0))
    data = raw.get_data()
    psd_data, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        average="mean",
        verbose=False,
    )  # (C, F)

    if calculate_average:
        psd_out = psd_data.mean(axis=0).astype(np.float32)  # (F,)
    else:
        psd_out = psd_data.astype(np.float32)  # (C, F)

    if normalize:
        psd_out = normalize_psd(psd_out)

    if return_freqs:
        return psd_out, freqs.astype(np.float32)
    return psd_out


def compute_psd_from_array(
    y: np.ndarray,
    *,
    sfreq: float,
    n_fft: int = PSD_CALCULATION_PARAMS["n_fft"],
    n_overlap: int = PSD_CALCULATION_PARAMS["n_overlap"],
    n_per_seg: int = PSD_CALCULATION_PARAMS["n_per_seg"],
    normalize: bool = True,
    return_freqs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for a 1‑D time series using the same bins as Raw PSD.

    Args:
        y: 1‑D array of samples (time domain)
        sfreq: Sampling frequency in Hz
        n_fft, n_overlap, n_per_seg: Welch parameters (match Raw defaults)
        normalize: If True, apply log10 + z‑score
        return_freqs: If True, also return frequency bins

    Returns:
        psd: (F,) float32. If return_freqs, also returns freqs (F,) float32.
    """
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim != 1:
        y_np = y_np.reshape(-1)
    psd_arr, freqs = mne.time_frequency.psd_array_welch(
        y_np[None, :],
        sfreq=float(sfreq),
        n_fft=int(n_fft),
        n_overlap=int(n_overlap),
        n_per_seg=int(n_per_seg),
        average="mean",
        verbose=False,
    )
    psd_vec = psd_arr[0].astype(np.float32)
    if normalize:
        psd_vec = normalize_psd(psd_vec)
    if return_freqs:
        return psd_vec, freqs.astype(np.float32)
    return psd_vec


# ------------------------------------------------------------------
# Method-specific time-domain preprocessing (no channel cleaning here)
# ------------------------------------------------------------------
def preprocess_time_domain_input(raw, *, target_sfreq: float = 128.0, segment_len_sec: int = 60) -> np.ndarray:
    """Resample, crop/pad, and z-score for time-domain models.
    Assumes channels already cleaned and ordered and typical bandpass done upstream.
    """
    x = raw.copy()
    x.load_data(verbose=False)
    sfreq_curr = float(x.info.get("sfreq", target_sfreq))
    tmax = min(segment_len_sec, x.times[-1])
    x.crop(tmin=0.0, tmax=tmax - 1.0 / sfreq_curr)
    if abs(sfreq_curr - target_sfreq) > 1e-3:
        x.resample(target_sfreq, npad="auto")
    data = x.get_data().astype(np.float32)  # (C, T)
    tgt_len = int(segment_len_sec * target_sfreq)
    if data.shape[1] < tgt_len:
        pad = tgt_len - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
        #print(f"ATTENTION: Padded {pad} samples to {tgt_len} samples")
    elif data.shape[1] > tgt_len:
        data = data[:, :tgt_len]
        print(f"ATTENTION: Truncated {data.shape[1]} samples to {tgt_len} samples")
    data = (data - data.mean()) / (data.std() + 1e-8)
    return data

def _to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[0]
    return x


def plot_psd(psd, freqs, path):
    x = _to_numpy_1d(psd)
    f = _to_numpy_1d(freqs)
    plt.figure()
    plt.plot(f, x)
    plt.xlabel("Hz")
    plt.ylabel("PSD (norm)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def plot_psd_comparison(psd1, psd2, freqs, path):
    x1 = _to_numpy_1d(psd1)
    x2 = _to_numpy_1d(psd2)
    f = _to_numpy_1d(freqs)
    plt.figure()
    plt.plot(f, x1, label="PSD 1")
    plt.plot(f, x2, label="PSD 2")
    plt.xlabel("Hz")
    plt.ylabel("PSD (norm)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()