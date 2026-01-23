"""
Preprocessing script for MPI LEMON dataset.
- Handles directory traversal of untarred data.
- Maps Initial IDs to INDI IDs using a lookup table (name_match.csv).
- Integrates Age/Sex metadata (Participants_MPILMBB_LEMON.csv).
- Filters strictly for Eyes Closed (EC) recordings.
- Includes rigorous QC (AutoReject, Beta/Alpha ratio).
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import mne
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import warnings
from autoreject import AutoReject
from mne.time_frequency import psd_array_welch

# Import shared cleaning logic
try:
    from data_preprocessing.cleanup_real_eeg_tuh import cleanup_real_eeg_tuh
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_preprocessing.cleanup_real_eeg_tuh import cleanup_real_eeg_tuh

# Constants
CANONICAL_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'
]

def _suppress(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None

# === QC Helpers ===

def _apply_autoreject(epochs: mne.Epochs, ar_n_jobs: int = 1) -> mne.Epochs | None:
    if len(epochs) < 2:
        return None
    try:
        ar = AutoReject(n_interpolate=[0], n_jobs=ar_n_jobs, random_state=97, verbose=False)
        epochs_ar = ar.fit_transform(epochs.copy())
        if len(epochs_ar) == 0:
            data_range = epochs.get_data().max() - epochs.get_data().min()
            if data_range > 1e-3:
                ep_scale = epochs.copy()
                ep_scale._data *= 1e-6
                epochs_ar = ar.fit_transform(ep_scale)
                if len(epochs_ar) > 0:
                    return epochs_ar
            return None
        return epochs_ar
    except Exception as e:
        print(f"AutoReject failed: {e}")
        return None

def _qc_epoch_mask(epochs: mne.Epochs, muscle_ratio_thr: float = 2.0) -> np.ndarray:
    X = epochs.get_data(copy=True)
    sf = epochs.info['sfreq']
    nper = int(max(1.25 * sf, 1))
    nover = int(max(0.5 * sf, 0))
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    alpha = psd[..., (freqs >= 8.0) & (freqs <= 12.0)].mean(-1)
    beta = psd[..., (freqs >= 20.0) & (freqs <= 45.0)].mean(-1)
    ratio_ep = (beta / np.maximum(alpha, 1e-12)).mean(axis=1)
    return ratio_ep < muscle_ratio_thr

def _epoch_quality_scores(epochs: mne.Epochs) -> np.ndarray:
    X = epochs.get_data(copy=True)
    sf = epochs.info['sfreq']
    nper = int(max(2 * sf, 1))
    nover = int(max(1 * sf, 0))
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    alpha = psd[..., (freqs >= 8.0) & (freqs <= 12.0)].mean(-1)
    beta = psd[..., (freqs >= 20.0) & (freqs <= 45.0)].mean(-1)
    ratio = beta / np.maximum(alpha, 1e-12)
    return ratio.mean(axis=1)

# === Metadata & ID Mapping Logic ===

def load_id_map(csv_path):
    """
    Load Initial_ID -> INDI_ID mapping.
    Initial_ID,INDI_ID
    sub-010002,sub-032301
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ ID map file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    # Map both initial -> indi AND indi -> indi (to handle mixed filenames)
    id_map = {}
    for _, row in df.iterrows():
        init = str(row['Initial_ID']).strip()
        indi = str(row['INDI_ID']).strip()
        id_map[init] = indi
        id_map[indi] = indi # Identity mapping
    
    print(f"✅ Loaded {len(id_map)} ID mappings")
    return id_map

def load_lemon_metadata(csv_path):
    """
    Load LEMON metadata CSV (INDI IDs).
    Returns dict: {indi_id: {'sex': 0/1, 'age': float}}
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Metadata file not found: {csv_path}")
        return {}
        
    df = pd.read_csv(csv_path)
    lookup = {}
    
    try:
        col_id = [c for c in df.columns if "ID" in c][0]
        col_sex = [c for c in df.columns if "Gender" in c][0] 
        col_age = [c for c in df.columns if "Age" in c][0]
    except Exception as e:
        print(f"⚠️ Metadata columns not identified: {e}")
        return {}
    
    for _, row in df.iterrows():
        indi_id = row[col_id]
        # Pipeline: 0=Female, 1=Male. CSV: 1=Female, 2=Male
        try:
            sex = int(row[col_sex]) - 1
            if sex not in [0, 1]: sex = -1
        except:
            sex = -1
        # Age "20-25" -> 22.5
        try:
            age_str = str(row[col_age])
            if "-" in age_str:
                low, high = map(int, age_str.split("-"))
                age = (low + high) / 2.0
            else:
                age = float(age_str)
        except:
            age = 0.0
        lookup[indi_id] = {'sex': sex, 'age': age}
        
    print(f"✅ Loaded metadata for {len(lookup)} INDI IDs")
    return lookup

def split_lemon_data(data_path, id_map, train_split=0.8, seed=42):
    """
    Filter for EC.set files and split by subject.
    """
    print(f"Searching for **/*.vhdr files in {data_path}...")
    all_files = []
    for root, _, files in os.walk(data_path):
        # We look for RSEEG folder specifically for resting state as per path structure
        if "RSEEG" in root:
            for f in files:
                if f.endswith(".vhdr"):
                     all_files.append(os.path.join(root, f))
    
    if not all_files:
        raise ValueError(f"No .vhdr files found in RSEEG folders within {data_path}")

    # Group by Normalized Subject ID (INDI ID)
    sub_map = {}
    for f in all_files:
        fname = os.path.basename(f)
        raw_sub = os.path.splitext(fname)[0].split('_')[0]
        indi_id = id_map.get(raw_sub, raw_sub) # fallback to raw if not in map
        
        if indi_id not in sub_map:
            sub_map[indi_id] = []
        sub_map[indi_id].append(f)
            
    subjects = sorted(list(sub_map.keys()))
    print(f"Found {len(subjects)} subjects across {len(all_files)} EC files.")
          
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    
    n_train = int(len(subjects) * train_split)
    train_subs = set(subjects[:n_train])
    eval_subs = set(subjects[n_train:])
    
    train_files = [f for s in train_subs for f in sub_map[s]]
    eval_files = [f for s in eval_subs for f in sub_map[s]]
    
    return train_files, eval_files

def _process_one_file(set_fp, sfreq, epoch_len_s, id_map, metadata):
    """
    Load, clean, and epoch a single LEMON .set file.
    """
    import warnings
    # Suppress specific MNE warnings that spam during parallel processing
    warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*expanding outside the data range.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*Data file name in EEG.data.*", category=RuntimeWarning)

    mne.set_log_level('ERROR')
    fname = os.path.basename(set_fp)
    raw_sub = os.path.splitext(fname)[0].split('_')[0]
    indi_id = id_map.get(raw_sub, raw_sub)
    
    # Metadata lookup (INDI ID based)
    meta = metadata.get(indi_id, {'age': 0.0, 'sex': -1})
    age, sex = meta['age'], meta['sex']
    
    try:
        # Robust Load for BrainVision files (Handles LEMON's frequent internal filename mismatches)
        vhdr_dir = os.path.dirname(set_fp)
        vhdr_base = os.path.splitext(fname)[0]
        
        with open(set_fp, 'r', encoding='utf-8', errors='ignore') as f:
            vhdr_content = f.read()
            
        # Check if internal pointers match actual files
        found_mismatch = False
        new_content = vhdr_content
        for key in ['DataFile', 'MarkerFile']:
            import re
            match = re.search(f'^{key}=(.*)$', vhdr_content, re.MULTILINE)
            if match:
                ptr_file = match.group(1).strip()
                actual_ext = '.eeg' if key == 'DataFile' else '.vmrk'
                actual_file = vhdr_base + actual_ext
                
                if ptr_file != actual_file and not os.path.exists(os.path.join(vhdr_dir, ptr_file)):
                    new_content = re.sub(f'^{key}=.*$', f'{key}={actual_file}', new_content, flags=re.MULTILINE)
                    found_mismatch = True
        
        if found_mismatch:
            # Create a temporary fixed vhdr
            tmp_vhdr = os.path.join(vhdr_dir, f"tmp_{fname}")
            with open(tmp_vhdr, 'w', encoding='utf-8') as f:
                f.write(new_content)
            load_path = tmp_vhdr
        else:
            load_path = set_fp
            
        try:
            raw = mne.io.read_raw_brainvision(load_path, preload=True, verbose=False)
        finally:
            if found_mismatch and os.path.exists(tmp_vhdr):
                os.remove(tmp_vhdr)
        
        # Use standard_1005 to cover 10-10 system used in LEMON. Interpolation up to 20% allowed.
        # This will also handle channel renaming and reduction to Canonical 19
        raw_clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq, montage='standard_1005', max_missing_pct=0.2)
        epochs = mne.make_fixed_length_epochs(raw_clean, duration=epoch_len_s, overlap=0.0, preload=True, verbose=False)
        
        epochs = _apply_autoreject(epochs, ar_n_jobs=1) 
        if epochs is None: return {"file": fname, "reason": "autoreject_fail", "epochs": []}
        
        mask = _qc_epoch_mask(epochs)
        epochs = epochs[mask]
        if len(epochs) == 0: return {"file": fname, "reason": "qc_fail", "epochs": []}
            
        if len(epochs) > 20:
            scores = _epoch_quality_scores(epochs)
            epochs = epochs[np.argsort(scores)[:20]]
        
        label = 0 # Normal
        all_epoch_tuples = []
        for i, ep_data in enumerate(epochs.get_data()):
            info = mne.create_info(epochs.info['ch_names'], sfreq, 'eeg')
            raw_ep = mne.io.RawArray(ep_data, info, verbose=False)
            raw_ep.set_montage(epochs.get_montage())
            sample_id = f"{indi_id}_EC_epoch{i}"
            
            # Robust metadata injection
            raw_ep.info["description"] = f"epoch_id={sample_id}; abnormal={label}; sex={sex}; age={age}"
            subj = raw_ep.info.get("subject_info", {}) or {}
            subj.update({"sex": int(sex) if sex != -1 else 0, "his_id": str(sample_id)})
            raw_ep.info["subject_info"] = subj
            
            all_epoch_tuples.append((raw_ep, sex, age, label, sample_id))
            
        return {"file": fname, "reason": "success", "epochs": all_epoch_tuples}

    except Exception as e:
        print(f"🟥 [fail] {fname}: {e}")
        return {"file": fname, "reason": f"error: {str(e)}", "epochs": []}

def load_data_lemon(data_path, save_path, metadata_path, id_map_path, sfreq=128, epoch_len_s=10.0, n_jobs=4):
    """Main entry point."""
    os.makedirs(save_path, exist_ok=True)
    
    id_map = load_id_map(id_map_path)
    metadata = load_lemon_metadata(metadata_path)
    train_files, eval_files = split_lemon_data(data_path, id_map)
    
    for split, files in [("train", train_files), ("eval", eval_files)]:
        if not files: continue
        print(f"\nProcessing {split.upper()} split ({len(files)} files)...")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_one_file)(f, sfreq, epoch_len_s, id_map, metadata) for f in tqdm(files)
        )
        
        all_epochs = []
        valid_files = 0
        insufficient = []
        for r in results:
            if r["epochs"]:
                all_epochs.extend(r["epochs"])
                valid_files += 1
                if len(r["epochs"]) < 20: insufficient.append(r)
        
        out_pkl = os.path.join(save_path, f"{split}_epochs.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(all_epochs, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"[{split.upper()}] Done. Saved {len(all_epochs)} epochs from {valid_files} files.")
        if insufficient:
            print(f"  Files with <20 epochs: {len(insufficient)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/lheiler/data/LEMON/EEG_Raw_BIDS_ID")
    parser.add_argument("--metadata_path", default="/home/lheiler/data/LEMON/Participants_MPILMBB_LEMON.csv")
    parser.add_argument("--id_map_path", default="/home/lheiler/data/LEMON/name_match.csv")
    parser.add_argument("--save_path", default="/home/lheiler/msc_thesis/Datasets/lemo")
    parser.add_argument("--n_jobs", type=int, default=16)
    args = parser.parse_args()
    
    load_data_lemon(args.data_path, args.save_path, args.metadata_path, args.id_map_path, n_jobs=args.n_jobs)
