import mne
import os

def load_data(data_path_train): #specifically for TUH EEG dataset
    
    t_data = []    
    for path in os.listdir(data_path_train):
        if path == ".DS_Store":
            continue
        
        for sub_path in os.listdir(os.path.join(data_path_train, path)):
            if sub_path == ".DS_Store":
                continue
            eeg_path = os.path.join(data_path_train, path, sub_path)
            #print("Loading training data from:", eeg_path)
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
            sex_code = raw.info['subject_info']['sex']
            #print if sexcode is not what we expect
            if sex_code not in [1, 2]:
                print(f"Sex code {sex_code} not what we expect")
            age = 0
            abn = 1 if path == "abnormal" else 0  # Abnormal
            # Attach a stable sample identifier based on relative path
            sample_id = f"{path}/{sub_path}"
            t_data.append((raw, sex_code, age, abn, sample_id))
    return t_data

