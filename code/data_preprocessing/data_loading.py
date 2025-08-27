import pickle
import os

def load_data(data_path_base, split="train"):
    """
    Load cleaned epoch data from pickle files.
    
    Parameters:
    -----------
    data_path_base : str
        Base path containing the pickle files (e.g., ~/thesis/Datasets/tuh-eeg-ab-clean)
    split : str
        Which split to load: "train" or "eval"
        
    Returns:
    --------
    List of tuples: (raw, g, a, ab, sample_id)
        raw: mne.Raw object
        g: gender (0=female, 1=male)
        a: age (always 0 for compatibility)
        ab: abnormal label (0=normal, 1=abnormal)
        sample_id: unique epoch identifier
    """
    pickle_file = os.path.join(data_path_base, f"{split}_epochs.pkl")
    
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    
    print(f"Loading {split} data from {pickle_file}...")
    
    with open(pickle_file, 'rb') as f:
        epoch_data = pickle.load(f)
    
    print(f"Loaded {len(epoch_data)} samples from {split} split")
    return epoch_data
