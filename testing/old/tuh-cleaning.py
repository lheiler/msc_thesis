import mne
import os
import numpy as np
import matplotlib.pyplot as plt

def rename_channel(name):
    """Rename EEG channel names to a standard format."""
    name = name.removeprefix("EEG ").removesuffix("-REF").strip().upper()
    
    if name.startswith("FP"):
        name = name.replace("FP", "Fp")
    if name == "FZ":
        name = "Fz"
    if name == "CZ":
        name = "Cz"
    if name == "PZ":
        name = "Pz"
    
    return name


raw = mne.io.read_raw_edf('/Users/lorenzheiler/small_dataset/eval/abnormal/aaaaaddm_s006_t000.edf', preload=True, verbose=False)


raw.crop(tmin=20.0, tmax=80.0)  # Crop to the 60s segment of the recording

# Housekeeping
raw.rename_channels({ch: rename_channel(ch) for ch in raw.ch_names})
raw.drop_channels(['T1', 'T2'])
raw.set_channel_types({'ROC': 'eog', 'LOC': 'eog',
                       'EKG1': 'ecg',
                       'PHOTIC': 'stim',
                       'IBI': 'misc', 'BURSTS': 'misc', 'SUPPR': 'misc'})
raw.set_montage('standard_1020')


# 1) Early resample to 250 Hz
raw.resample(250, npad="auto")

# 2) High-pass + notch
raw.filter(l_freq=1., h_freq=None)
raw.notch_filter(freqs=[50, 100])

# 3) ICA
ica = mne.preprocessing.ICA(n_components=19, method='infomax', random_state=97)
    # Select all EEG, EOG and ECG channels for ICA
picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, stim=False, misc=False)
ica.fit(raw, picks=picks)

# ica.plot_components(inst=raw, ch_type='eeg', title='ICA components', show=True)
# ica.plot_sources(inst=raw, title='ICA sources', show=True)
# plt.show()

ica.exclude += ica.find_bads_eog(raw, ch_name=['ROC', 'LOC'])[0]
ica.exclude += ica.find_bads_ecg(raw, ch_name='EKG1')[0]
raw_clean = ica.apply(raw.copy())

# 4) Drop helpers, rereference, final filters
raw_clean.drop_channels(['ROC', 'LOC', 'EKG1'])
raw_clean.set_eeg_reference('average', projection=False)
raw_clean.filter(l_freq=None, h_freq=40.)

# 5) (Optional) final resample to 128 Hz
raw_clean.resample(128, npad="auto")


print(raw_clean.ch_names)

#return raw_clean
