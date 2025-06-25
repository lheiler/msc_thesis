import torch
from torch.utils.data import DataLoader
import mne
import os
import numpy as np



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