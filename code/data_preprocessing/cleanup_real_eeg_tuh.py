
import mne
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os 

os.environ["JOBLIB_VERBOSE"] = "0" # Suppress joblib warnings


date = 2147483

def cleanup_real_eeg_tuh(raw: mne.io.BaseRaw, sfreq, montage='standard_1020', picks=None, ref_channels='average', segment_length=60) -> mne.io.BaseRaw:
    
    mne.set_log_level('WARNING')  # Suppress MNE warnings
    
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

    
    raw.crop(tmin=(raw.times[-1] / 2), tmax=segment_length+(raw.times[-1] / 2))  # Crop to the 60s segment of the recording
    # Housekeeping
    raw.rename_channels({ch: rename_channel(ch) for ch in raw.ch_names})
    raw.drop_channels(['T1', 'T2', 'EMG', '26', '27', '28', '29', '30'], on_missing='ignore')
    
    rocloc = raw.ch_names.count('ROC') > 0 and raw.ch_names.count('LOC') > 0
    if rocloc:
        raw.set_channel_types({'ROC': 'eog', 'LOC': 'eog'})
    
    if raw.ch_names.count('EKG1') > 0:
        raw.set_channel_types({'EKG1': 'ecg'})
        
    if raw.ch_names.count('PHOTIC') > 0:
        raw.set_channel_types({'PHOTIC': 'stim'})
        
    if raw.ch_names.count('IBI') > 0 or raw.ch_names.count('BURSTS') > 0 or raw.ch_names.count('SUPPR') > 0:
        raw.set_channel_types({'IBI': 'misc', 'BURSTS': 'misc', 'SUPPR': 'misc'}, on_unit_change="ignore", verbose=False)
        
    
    raw.set_montage(montage, on_missing='ignore', verbose=False)  # Set montage to standard 10-20 system


    # 1) Early resample to 250 Hz
    raw.resample(256, npad="auto")

    # 2) High-pass + notch
    raw.filter(l_freq=1., h_freq=None, verbose=False)
    #raw.notch_filter(freqs=[60])

    # 3) ICA
    ica = mne.preprocessing.ICA(n_components=19, method='infomax', random_state=97, verbose=False)
        # Select all EEG, EOG and ECG channels for ICA
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=False, stim=False, misc=False)
    ica.fit(raw, picks=picks, verbose=False)

    # ica.plot_components(inst=raw, ch_type='eeg', title='ICA components', show=True)
    # ica.plot_sources(inst=raw, title='ICA sources', show=True)
    # plt.show()

    if rocloc:
        ica.exclude += ica.find_bads_eog(raw, ch_name=['ROC', 'LOC'], verbose=False)[0]
    else:
        ica.exclude += ica.find_bads_eog(raw, ch_name='Fp1', verbose=False)[0]
    if raw.ch_names.count('EKG1') > 0:
        ica.exclude += ica.find_bads_ecg(raw, ch_name='EKG1', verbose=False)[0]
    
    raw_clean = ica.apply(raw.copy(), verbose=False)

    # 4) Drop helpers, rereference, final filters    
    raw_clean.drop_channels(['ROC', 'LOC', 'EKG1'], on_missing='ignore')
    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    raw_clean.filter(l_freq=None, h_freq=40., verbose=False)

    # 5) (Optional) final resample to 128 Hz
    raw_clean.resample(sfreq, npad="auto")

    raw_clean.set_meas_date(date)  # Set measurement date to a fixed value for consistency
    

    return raw_clean




def load_data(data_path_train, data_path_eval, save_path, sfreq=128): #specifically for TUH EEG dataset
    
    if data_path_train.endswith("/"): data_path_train = data_path_train[:-1]
    if data_path_eval.endswith("/"): data_path_eval = data_path_eval[:-1]
        
    save_path_train = os.path.join(save_path, "train")
    save_path_eval = os.path.join(save_path, "eval")
        
    for path in os.listdir(data_path_train):
        if path == ".DS_Store":
            continue
        
        for sub_path in os.listdir(os.path.join(data_path_train, path)):
            print(sub_path)
            if not sub_path.endswith(".edf"):
                continue
            eeg_path = os.path.join(data_path_train, path, sub_path)
            print("Cleaning training data from:", eeg_path)
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
            # clean.plot(scalings='auto')
            # plt.show()
            
            # Save cleaned data
            save_file_path = os.path.join(save_path_train, path, sub_path)
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)        
            save_file_path = save_file_path.replace(".edf", "_eeg.fif")
            clean.save(save_file_path, overwrite=True, verbose=False)  
            
    # Create DataLoader for training data
                
    
    for path in os.listdir(data_path_eval):
        if path == ".DS_Store":
            continue
        for sub_path in os.listdir(os.path.join(data_path_eval, path)):
            if not sub_path.endswith(".edf"):
                continue
            eeg_path = os.path.join(data_path_eval, path, sub_path)
            print("Cleaning evaluation data from:", eeg_path)
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
            # Save cleaned data in edf format
            save_file_path = os.path.join(save_path_eval, path, sub_path)
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            save_file_path = save_file_path.replace(".edf", "_eeg.fif")
            clean.save(save_file_path, overwrite=True, verbose=False)  
            
    return






if __name__ == "__main__":
    # Example usage
    data_path_train = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab/edf/train"  # Specify the path to your training data
    data_path_eval = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab/edf/eval"  # Specify the path to your evaluation data

    save_path = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean"  # Specify the path to save cleaned data
    


    sfreq = 128  # Specify desired sampling frequency after preprocessing
    clean_data = True  # Set to True if you want to clean the data
    
    load_data(data_path_train, data_path_eval, save_path, sfreq)
    
    print("Data cleaning done.")
