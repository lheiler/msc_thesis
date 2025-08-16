import json
import numpy as np
import torch
import mne

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
}


def clean_raw_eeg(raw, segment_length=60.0):
    """Drop aux channels (A1/A2) and pick the standard 19 EEG channels.
    Mirrors the prior inline cleaning logic used in extractors.
    """
    if 'A1' in raw.ch_names:
        raw.drop_channels(['A1'])
    if 'A2' in raw.ch_names:
        raw.drop_channels(['A2'])
    raw.pick_channels(STANDARD_EEG_CHANNELS)
    raw.crop(tmin=0.0, tmax=segment_length-1.0/raw.info['sfreq'])
    raw.filter(3.0, 45.0, fir_design="firwin", verbose=False)
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
        
def normalize_psd(psd):
    log_psd = np.log10(psd + 1e-12)
    return (log_psd - log_psd.mean()) / log_psd.std()




def compute_psd_from_raw(raw, *, n_fft=256, n_overlap=128, n_per_seg=256,  calculate_average=False) -> np.ndarray:
    """Compute average PSD vector across EEG channels.
    If sfreq is provided, use psd_array_welch on the numpy data; otherwise use Raw.compute_psd with fmin/fmax.
    Returns float32 vector; applies log10 if log=True.
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    psd_data, _ = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, average="mean", verbose=False
    )
    if calculate_average:
        psd_data = np.mean(psd_data, axis=0).astype(np.float32)
    else:
        psd_data = np.array([psd_data]).astype(np.float32)
     
    psd_data = normalize_psd(psd_data)
    
    print(psd_data.shape)
    return psd_data


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
        print(f"ATTENTION: Padded {pad} samples to {tgt_len} samples")
    elif data.shape[1] > tgt_len:
        data = data[:, :tgt_len]
        print(f"ATTENTION: Truncated {data.shape[1]} samples to {tgt_len} samples")
    data = (data - data.mean()) / (data.std() + 1e-8)
    return data
