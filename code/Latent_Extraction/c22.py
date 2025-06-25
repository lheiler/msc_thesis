# function to extract features from Raw instance through catch22 method
import pycatch22
import numpy as np
import mne

def extract_values(features):
    """Extract values from Catch22 feature output dict."""
    return np.array(features['values'], dtype=np.float32)


def extract_c22_psd(x: mne.io.Raw) -> np.ndarray:
    """    Extract catch22 features from a batch of EEG data.   
    """
    print("Extracting catch22 features of psd...")
    
    # drop A1 and A2 channels if they exist
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])
    
    # array of only eeg channel data
    x = x.get_data(picks='eeg')
    
    #compute the average power spectral density (PSD) of the EEG data
    psd_data, _ = mne.time_frequency.psd_array_welch(x, sfreq=128, n_fft=2048, n_overlap=1024, n_per_seg=2048, average='mean', verbose=False)
    psd_data = np.mean(psd_data, axis=0)  # Average across channels
    psd_data = psd_data.astype(np.float32)  # Convert to float32
    # Extract catch22 features from the PSD
    features = pycatch22.catch22_all(psd_data)
    
    return extract_values(features)

def extract_c22(x: mne.io.Raw) -> np.ndarray:
    """Extract catch22 features from a batch of EEG data."""
    print("Extracting catch22 features...")
    
    if 'A1' in x.ch_names:
        x.drop_channels(['A1'])
    if 'A2' in x.ch_names:
        x.drop_channels(['A2'])
    
    # Extract catch22 features for raw signal of each channel in the EEG dat
    all_features = []
    x = x.get_data(picks='eeg')  # Only consider EEG channels
    
    for i in range(x.shape[0]):
        channel_data = x[i, :]  # Get data for the i-th channel
        features = pycatch22.catch22_all(channel_data)
        all_features.append(extract_values(features))
    all_features = np.concatenate(all_features, axis=0)  # Concatenate features from
    return all_features
