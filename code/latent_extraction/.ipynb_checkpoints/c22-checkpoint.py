# function to extract features from Raw instance through catch22 method
import pycatch22
import numpy as np
import mne
from utils.util import STANDARD_EEG_CHANNELS

def extract_values(features):
    """Extract values from Catch22 feature output dict."""
    return np.array(features['values'], dtype=np.float32)



def extract_c22(x: mne.io.Raw) -> np.ndarray:
    """Extract catch22 features from a batch of EEG data."""
    all_features = []
    for ch in STANDARD_EEG_CHANNELS:
        if ch in x.ch_names:
            ch_data = x.copy().pick([ch]).get_data()[0]
            features = pycatch22.catch22_all(ch_data)
            all_features.append(extract_values(features))
        else:
            print(f"Missing channel: {ch} — padding with zeros")
            all_features.append(np.zeros(22, dtype=np.float32))  # 22 zero features
    return np.concatenate(all_features, axis=0)
