# function to extract features from Raw instance through catch22 method
import pycatch22
import numpy as np
import mne
from utils.util import clean_raw_eeg, STANDARD_EEG_CHANNELS

def extract_values(features):
    """Extract values from Catch22 feature output dict."""
    return np.array(features['values'], dtype=np.float32)


def extract_c22_psd(x: mne.io.Raw) -> np.ndarray:
    """    Extract catch22 features from a batch of EEG data.   
    """
    print("Extracting catch22 features of psd...")
    
    # Kept for backwards compatibility: compute PSD internally
    x = clean_raw_eeg(x)
    data = x.get_data(picks='eeg')
    psd_data, _ = mne.time_frequency.psd_array_welch(
        data, sfreq=128, n_fft=2048, n_overlap=1024, n_per_seg=2048, average='mean', verbose=False
    )
    psd_avg = np.mean(psd_data, axis=0).astype(np.float32)
    return extract_c22_from_psd(psd_avg)


def extract_c22_from_psd(psd_vector: np.ndarray) -> np.ndarray:
    """Extract catch22 features given a precomputed PSD vector (float32)."""
    features = pycatch22.catch22_all(psd_vector.astype(np.float32))
    return extract_values(features)

def extract_c22(x: mne.io.Raw) -> np.ndarray:
    """Extract catch22 features from a batch of EEG data."""
    print("Extracting catch22 features...")
    
    # Standardize channels
    x = clean_raw_eeg(x)

    # Extract catch22 features for raw signal of each channel in the EEG data
    all_features = []
    # Extract features channel-by-channel in the fixed order
    for ch in STANDARD_EEG_CHANNELS:
        if ch in x.ch_names:
            ch_data = x.copy().pick_channels([ch]).get_data()[0]
            features = pycatch22.catch22_all(ch_data)
            all_features.append(extract_values(features))
        else:
            print(f"Missing channel: {ch} — padding with zeros")
            all_features.append(np.zeros(22, dtype=np.float32))  # 22 zero features
    return np.concatenate(all_features, axis=0)
