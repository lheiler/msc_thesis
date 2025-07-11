import torch
from torch.utils.data import DataLoader
import mne
import os
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings



def load_data(data_path_train): #specifically for TUH EEG dataset
    
    t_data = []    
    for path in os.listdir(data_path_train):
        if path == ".DS_Store":
            continue
        
        for sub_path in os.listdir(os.path.join(data_path_train, path)):
            if sub_path == ".DS_Store":
                continue
            eeg_path = os.path.join(data_path_train, path, sub_path)
            print("Loading training data from:", eeg_path)
            raw = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False)
            sex_code = raw.info['subject_info']['sex']
            age = 0
            abn = 1 if path == "abnormal" else 0  # Abnormal
            t_data.append((raw, sex_code, age, abn))

    return t_data


def load_data_harvard(data_path, eval_split=0.2):
    """
    Load cleaned Harvard EEG data from BIDS format and extract subject-level metadata.

    Returns:
        train_array (list of tuples): (raw, sex_code, age, abn)
        eval_array (list of tuples): (raw, sex_code, age, abn)
    """
    
    ### IMPORTANT: CURRENTLY SKIPS ALL NON rEEG TASKS
    
    
    data_path = Path(data_path)
    subject_dirs = sorted([p for p in data_path.glob("sub-*") if p.is_dir()])

    subject_sessions = []
    for subj_dir in subject_dirs:
        for ses_dir in subj_dir.glob("ses-*"):
            subject = subj_dir.name.replace("sub-", "")
            session = ses_dir.name.replace("ses-", "")
            subject_sessions.append((subject, session))
            #print(f"Found subject-session pair: {subject}, {session}")

    # Split subject-session pairs into train/eval
    train_pairs, eval_pairs = train_test_split(subject_sessions, test_size=eval_split, random_state=42, shuffle=True)

    def process(pairs):
        result = []
        for i,(subject, session) in enumerate(pairs):
            try:
                
                bids_path = BIDSPath(subject=subject,
                     session=session,
                     suffix='eeg',
                     extension='.vhdr',
                     datatype='eeg',
                     root=data_path)
                
                with warnings.catch_warnings():
                    print(f"Reading BIDS path: {bids_path.fpath}")
                    #warnings.simplefilter("ignore", category=RuntimeWarning)
                    raw = read_raw_bids(bids_path, verbose=False)
                    
                raw.load_data()
                sfreq = raw.info['sfreq']
                
                # crop data to exactly sfreq*60 samples
                if raw.n_times > sfreq * 60:
                    raw.crop(tmax=60 - 1/sfreq)  # 60 seconds at 128 Hz

                # Read metadata from sidecar
                json_path = bids_path.copy().update(suffix='eeg', extension='.json').fpath
                with open(json_path, 'r') as f:
                    sidecar = json.load(f)

                meta = sidecar.get("HEEDB_meta", {})
                sex_code = int(1 if meta.get("SexDSC") == "Male" else (2 if meta.get("SexDSC") == "Female" else 0))
                age = float(meta.get("AgeAtVisit", -1))         # adapt if the key is different
                abn = int(0 if meta.get("abnormal", 0)== "nan" else 1)       # or another appropriate key
                
                print(i,(raw, sex_code, age, abn))
                result.append((raw, sex_code, age, abn))

            except Exception as e:
                #print(f"⚠️ Skipping {subject}, {session}: {e}")
                continue

        return result

    train_array = process(train_pairs)
    eval_array = process(eval_pairs)
    print(f"Loaded {len(train_array)} training samples and {len(eval_array)} evaluation samples.")
    return train_array, eval_array

if __name__ == "__main__":
    """
    Main function to load and preprocess data.
    """

    # print(os.listdir())
    # data_path_train = "/Users/lorenzheiler/small_dataset/train"  # Specify the path to your training data
    # data_path_eval = "/Users/lorenzheiler/small_dataset/eval"  # Specify the path to your evaluation data

    # sfreq = 128  # Specify desired sampling frequency after preprocessing
    # clean_data = True  # Set to True if you want to clean the data
    # batch_size = 32  # Specify the batch size for DataLoader
    
    # train_loader, eval_loader = load_data(data_path_train, data_path_eval, sfreq, clean_data, batch_size)
    
    # print(f"Loaded {len(train_loader.dataset)} training samples and {len(eval_loader.dataset)} evaluation samples.")