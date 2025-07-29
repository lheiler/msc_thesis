import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.preprocessing import ICA

# Step 1: Download and load the data
subject = 1
runs = [7]  # Resting state
raw_fnames = eegbci.load_data(subject, runs)
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)

# Step 2: Preprocessing
raw.filter(l_freq=1.0, h_freq=40.0)
raw.set_eeg_reference(ref_channels='average', projection=False)

raw.plot(scalings='auto', title='Raw Data', show=True)
plt.show()

# -------------- Step 2.1: Remove bad channels --------------

# -------------- Step 2.5: Fixing channel names --------------
raw.rename_channels(lambda name: name.upper().replace('.', '').replace('Z', 'z'))
def fix_channel_name(name):
    name = name.replace('.', '').replace('..', '')
    if name.startswith('FP'):
        name = 'Fp' + name[2:]
    return name

raw.rename_channels(fix_channel_name)
raw.set_montage('standard_1005')
# --------------------------------------------------

# Step 3: ICA
ica = ICA(n_components=15, random_state=97, max_iter=800)
ica.fit(raw, picks='eeg')  # Pick EEG channels only

eog_inds, scores = ica.find_bads_eog(raw, ch_name='Fpz')  
print(eog_inds)  # Indices of components that correlate with EOG

ica.plot_components(inst=raw, ch_type='eeg', title='ICA components', show=True)
ica.plot_sources(inst=raw, title='ICA sources', show=True)
plt.show()

# Step 4: Remove EOG components
ica.exclude = eog_inds
ica.apply(raw)  # removes the blink component
#plot adjusted raw data (took away the blinking component)
raw.plot(scalings='auto', title='Adjusted Raw Data', show=True)

plt.show()