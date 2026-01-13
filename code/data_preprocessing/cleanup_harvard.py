"""
Cleaned EEG preprocessing pipeline for TUH dataset.
Refactored for conciseness while preserving all logic.
"""

import mne
import os
import numpy as np
import pandas as pd
import pickle
from autoreject import AutoReject
from mne.time_frequency import psd_array_welch
from typing import List, Iterable, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import glob
from collections import Counter
import gc

# === Constants ===
EDGE_PAD_S: float = 0.5
HP_HZ: float = 0.5
LP_HZ: float = 45.0
ICA_N_COMPONENTS: float = 0.99
METADATA_DIR = "/rds/general/user/lrh24/ephemeral/harvard/metadata"

CANONICAL_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'
]

EXTRA_NONCANON = {
    'T1','T2','26','27','28','29','30','31','32','A1','A2',
    'C3P','C4P','OZ','SP1','SP2','PG1','PG2','IBI','BURSTS','SUPPR'
}

os.environ["JOBLIB_VERBOSE"] = "0"


# === Helper Functions ===
def _is_cardiac(name: str) -> bool:
    """Check if channel name indicates cardiac signal."""
    return any(x in name.upper() for x in ('ECG', 'EKG', 'PULSE'))


def _suppress(fn, *args, **kwargs):
    """Suppress exceptions and return None on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _safe_set_montage(inst, montage='standard_1020'):
    """Set montage safely, print warning on failure."""
    try:
        inst.set_montage(montage, verbose=False)
    except Exception:
        print("🛑[canonical] standard_1020 montage not found")


def _fmt_dur(sec: float) -> str:
    """Format seconds with 2 decimals, fallback to str on error."""
    try:
        return f"{sec:.2f}s"
    except:
        return str(sec)


def _event_samples(ev) -> np.ndarray:
    """Normalize event outputs to 1-D sample indices."""
    if ev is None:
        return np.array([], dtype=int)
    if isinstance(ev, tuple):
        ev = ev[0]
    
    arr = np.asarray(ev)
    if arr.ndim == 0:
        return np.array([], dtype=int)
    elif arr.ndim == 1:
        return arr.astype(int)
    elif arr.ndim == 2:
        return arr[:, 0].astype(int)
    else:
        return np.array([], dtype=int)


def _mad_clip_inplace(raw_hp: mne.io.BaseRaw, picks: Iterable[int], limit: float = 20.0) -> None:
    """Clip outliers using MAD for ICA stability."""
    X = raw_hp.get_data(picks=picks)
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + 1e-12
    raw_hp._data[picks, :] = np.clip(X, med - limit * mad, med + limit * mad)


# === Channel Processing ===
def rename_channel(name: str) -> str:
    """Canonicalize TUH channel names."""
    name = name.removeprefix("EEG ").removesuffix("-REF").strip().upper()
    
    # Normalize case
    if name.startswith("FP"):
        name = name.replace("FP", "Fp")
    if name in ("FZ", "CZ", "PZ"):
        name = name.replace("Z", "z")
    
    # Legacy mapping
    legacy_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    return legacy_map.get(name, name)


def trim_zero_edges(raw: mne.io.BaseRaw, eps: float = 0.0, min_keep_sec: float = 1.0, verbose=print) -> None:
    """Trim leading/trailing regions where all EEG samples are near zero."""
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    if not len(picks):
        return
    
    data = raw.get_data(picks=picks)
    zero_mask = np.all(np.abs(data) <= eps, axis=0)
    n, sf = zero_mask.size, raw.info['sfreq']
    
    nz_front = np.argmax(~zero_mask) if zero_mask[0] else 0
    if nz_front == 0 and zero_mask[0]:
        return
    
    nz_back = n - 1 - np.argmax(~zero_mask[::-1]) if zero_mask[-1] else n - 1
    tmin, tmax = nz_front / sf, nz_back / sf
    
    if tmax - tmin >= max(min_keep_sec, 1.0 / sf):
        raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        verbose(f"[trim] zero edges to [{_fmt_dur(tmin)}, {_fmt_dur(tmax)}]")


def conform_to_canonical(raw: mne.io.BaseRaw, canonical: List[str]) -> mne.io.BaseRaw:
    """Pick and order channels to canonical set; raise if any are missing."""
    missing = [ch for ch in canonical if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"🛑 [canonical] missing canonical channels: {missing}")
    
    raw.pick(canonical, verbose=False)
    _safe_set_montage(raw, 'standard_1020')
    return raw


# === Artifact Detection ===
def annotate_artifacts(raw: mne.io.BaseRaw, verbose=print) -> None:
    """Annotate artifact segments as BAD_* without altering data."""
    sfreq = raw.info['sfreq']
    orig_time = getattr(raw.annotations, 'orig_time', None)
    new_anns = []
    
    # Muscle (EEG-wide z-score)
    mus = _suppress(mne.preprocessing.annotate_muscle_zscore, raw, ch_type='eeg', threshold=4.0, min_length_good=0.2)
    if mus is not None and len(mus[0]):
        new_anns.append(mne.Annotations(mus[0].onset, mus[0].duration, mus[0].description, orig_time=orig_time))
    
    # EMG channel bursts
    if 'EMG' in raw.ch_names:
        def _emg_ann():
            emg = raw.copy()
            emg.pick('EMG')
            emg.filter(20., 100., verbose=False)
            x = emg.get_data()[0]
            w = int(max(0.2 * sfreq, 1))
            rms = np.sqrt(np.convolve(x * x, np.ones(w) / w, mode='same'))
            med = np.median(rms)
            mad = np.median(np.abs(rms - med)) + 1e-12
            above = rms > (med + 5 * mad)
            
            trans = np.diff(np.concatenate(([0], above.astype(int), [0])))
            starts, ends = np.where(trans == 1)[0], np.where(trans == -1)[0]
            on, du = starts / sfreq, (ends - starts) / sfreq
            keep = du >= 0.2
            
            if keep.any():
                return mne.Annotations(on[keep], du[keep], ['BAD_muscle_emg'] * int(keep.sum()), orig_time=orig_time)
        
        ann = _suppress(_emg_ann)
        if ann is not None:
            new_anns.append(ann)
    
    # Photic stimulation
    if 'PHOTIC' in raw.ch_names:
        def _safe_find_events():
            events = mne.find_events(raw, stim_channel='PHOTIC', shortest_event=1, verbose=False)
            return events.astype(int) if events is not None and len(events) > 0 else None
        
        ev = _suppress(_safe_find_events)
        smp = _event_samples(ev)
        if smp.size:
            on = smp / sfreq
            new_anns.append(mne.Annotations(on, [2.0] * len(on), ['BAD_photic'] * len(on), orig_time=orig_time))
    
    # Technician-derived BAD_*
    if len(raw.annotations):
        bad_keys = ('hyper', 'hv', 'photic', 'test', 'artifact', 'movement', 'talk')
        sel = [i for i, d in enumerate(raw.annotations.description) 
               if any(k in str(d).lower() for k in bad_keys)]
        
        if sel:
            starts = [raw.annotations.onset[i] for i in sel]
            durs = [max(float(raw.annotations.duration[i]), 1.0) for i in sel]
            descs = [f"BAD_{raw.annotations.description[i]}" if not str(raw.annotations.description[i]).lower().startswith('bad_') 
                    else raw.annotations.description[i] for i in sel]
            new_anns.append(mne.Annotations(starts, durs, descs, orig_time=orig_time))
    
    # Merge and summarize
    for ann in new_anns:
        raw.set_annotations(raw.annotations + ann)
    
    c = Counter([d for d in raw.annotations.description if str(d).upper().startswith('BAD_')])
    if c:
        verbose("[annots] " + ", ".join([f"{k}:{v}" for k, v in sorted(c.items())]))


# === Main Cleaning Function ===
def cleanup_real_eeg_tuh(raw: mne.io.BaseRaw, sfreq: float, montage: str = 'standard_1020', verbose: bool = False) -> mne.io.BaseRaw:
    """End-to-end cleaning for a TUH EEG recording."""
    log = print if verbose else (lambda *a, **k: None)
    
    # Initial setup
    log(f"[input] {len(raw.ch_names)} ch, sfreq={raw.info.get('sfreq','NA')}, dur={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")
    
    # Channel processing
    raw.rename_channels({ch: rename_channel(ch) for ch in raw.ch_names})
    raw.drop_channels(list(EXTRA_NONCANON), on_missing='ignore')
    
    # Set channel types
    channel_types = {}
    if raw.ch_names.count('ROC') and raw.ch_names.count('LOC'):
        channel_types.update({'ROC': 'eog', 'LOC': 'eog'})
    
    channel_types.update({ch: 'ecg' for ch in raw.ch_names if _is_cardiac(ch)})
    channel_types.update({ch: 'emg' for ch in raw.ch_names if 'EMG' in ch.upper()})
    channel_types.update({ch: 'stim' for ch in raw.ch_names if 'PHOT' in ch.upper()})
    
    if channel_types:
        raw.set_channel_types(channel_types)
    
    # Montage and trimming
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    trim_zero_edges(raw, verbose=log)
    log(f"[mont+trim] {len(raw.ch_names)} ch, dur={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")
    
    # Filtering and ICA
    raw.resample(256, npad="auto")
    raw.filter(l_freq=HP_HZ, h_freq=LP_HZ, verbose=False)
    log(f"[bp] {HP_HZ}-{LP_HZ} Hz")
    
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    
    if ICA_N_COMPONENTS and picks_eeg.size:
        log(f"[ICA] fitting (picard, n={ICA_N_COMPONENTS}, decim=5)")
        ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method='picard', random_state=97, verbose=False)
        
        # Prepare data for ICA
        # OPTIMIZATION: Use float32 and try to minimize copies
        raw_hp = raw.copy()
        raw_hp.filter(1.0, None, verbose=False)
        X = raw_hp.get_data(picks=picks_eeg)
        
        # Check for near-zero variance channels
        stds = X.std(axis=1)
        good = stds > 1e-12
        if not np.all(good):
            bad_names = [raw.ch_names[picks_eeg[i]] for i in np.where(~good)[0]]
            print(f"🟥 [ICA] excluding near-zero-variance channels: {bad_names}")
            picks_eeg = picks_eeg[good]
        
        if len(picks_eeg):
            _mad_clip_inplace(raw_hp, picks_eeg, limit=20.0)
            ica.fit(raw_hp, picks=picks_eeg, decim=5, verbose=False)
            
            # Find EOG components
            if raw.ch_names.count('ROC') and raw.ch_names.count('LOC'):
                eog_idx, _ = ica.find_bads_eog(raw, ch_name=['ROC', 'LOC'], verbose=False)
                ica.exclude += eog_idx
                log(f"[ICA] EOG ROC/LOC: {list(eog_idx)}")
            else:
                eog_cands = [ch for ch in ['EOG', 'VEOG', 'HEOG', 'Fp1', 'Fp2'] if ch in raw.ch_names]
                if eog_cands:
                    eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_cands, verbose=False)
                    ica.exclude += eog_idx
                    log(f"[ICA] EOG {eog_cands}: {list(eog_idx)}")
            
            # Find ECG components
            ecg_cands = [ch for ch in raw.ch_names if _is_cardiac(ch)]
            if ecg_cands:
                ecg_idx, _ = ica.find_bads_ecg(raw, ch_name=ecg_cands[0], verbose=False)
                ica.exclude += ecg_idx
                log(f"[ICA] ECG {ecg_cands[0]}: {list(ecg_idx)}")
            
            log(f"[ICA] exclude: {sorted(set(ica.exclude))}")
            # OPTIMIZATION: Apply in-place to save memory
            ica.apply(raw, verbose=False)
            raw_clean = raw
            log("[ICA] applied")
        else:
            raw_clean = raw.copy()
    else:
        log("[ICA] skipped")
        raw_clean = raw.copy()
    
    # Final processing
    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    log("[ref] average")
    
    annotate_artifacts(raw_clean, verbose=log)
    
    # Drop helper channels
    cardiac = [ch for ch in raw_clean.ch_names if _is_cardiac(ch)]
    eog = [ch for ch in ['ROC', 'LOC', 'EOG', 'VEOG', 'HEOG'] if ch in raw_clean.ch_names]
    if cardiac or eog:
        raw_clean.drop_channels(eog + cardiac, on_missing='ignore')
        log(f"[drop helpers] {len(raw_clean.ch_names)} ch")
    
    # Final resampling and edge padding
    raw_clean.resample(sfreq, npad="auto")
    dur = raw_clean.times[-1] if len(raw_clean.times) else 0.0
    
    if dur > 2.0:
        edge = EDGE_PAD_S
        orig_time = getattr(raw_clean.annotations, 'orig_time', None)
        edge_anns = mne.Annotations([0, dur - edge], [edge, edge], ['BAD_edge', 'BAD_edge'], orig_time=orig_time)
        raw_clean.set_annotations(raw_clean.annotations + edge_anns)
        log(f"[edge] marked {EDGE_PAD_S}s edges")
    
    raw_clean.set_meas_date(None)
    
    # Ensure canonical channels
    missing = [ch for ch in CANONICAL_19 if ch not in raw_clean.ch_names]
    if missing:
        print(f"🟥 [fail] missing canonical channels before pick: {missing}")
        raise ValueError("missing canonical channels before pick")
    
    conform_to_canonical(raw_clean, CANONICAL_19)
    
    return raw_clean


# === Epoching and Quality Control ===
def _make_fixed_length_epochs(raw: mne.io.BaseRaw, epoch_len: float) -> mne.Epochs:
    """Segment into fixed, non-overlapping epochs."""
    return mne.make_fixed_length_epochs(
        raw, duration=epoch_len, overlap=0.0, preload=True,
        verbose=False, reject_by_annotation=True
    )


def _apply_autoreject(epochs: mne.Epochs, ar_n_jobs: int = -1) -> mne.Epochs | None:
    """Apply AutoReject with fallback strategies."""
    n_ep = len(epochs)
    if n_ep < 2:
        print("🟥 [autoreject] skipping: insufficient epochs (n<2)")
        return None
    
    def _try_autoreject(ep_copy):
        ar = AutoReject(n_interpolate=[0], n_jobs=ar_n_jobs, random_state=97, verbose=False)
        return ar.fit_transform(ep_copy)
    
    try:
        # First attempt
        epochs_ar = _try_autoreject(epochs.copy())
        
        if len(epochs_ar) == 0:
            print("🟠 [autoreject] rejected all epochs; trying with µV→V scaling")
            data_range = epochs.get_data().max() - epochs.get_data().min()
            
            if data_range > 1e-3:  # Likely in µV
                epochs_scaled = epochs.copy()
                epochs_scaled._data *= 1e-6
                epochs_ar = _try_autoreject(epochs_scaled)
                
                if len(epochs_ar) > 0:
                    print(f"[autoreject] µV scaling worked; n_epochs_out={len(epochs_ar)}")
                    return epochs_ar
                else:
                    print("🟥 [autoreject] all epochs rejected; discarding sample")
                    return None
            else:
                print("🟥 [autoreject] all epochs rejected; discarding sample")
                return None
        else:
            return epochs_ar
            
    except Exception as e:
        print(f"🟥 [autoreject] failed ({e}); discarding sample")
        return None


def _qc_epoch_mask(epochs: mne.Epochs, muscle_ratio_thr: float = 2.0) -> np.ndarray:
    """Return boolean mask of epochs passing QC using beta/alpha power ratio."""
    X = epochs.get_data(copy=True)
    sf = epochs.info['sfreq']
    nper = int(max(1.25 * sf, 1))
    nover = int(max(0.5 * sf, 0))
    
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    
    alpha = psd[..., (freqs >= 8.0) & (freqs <= 12.0)].mean(-1)
    beta = psd[..., (freqs >= 20.0) & (freqs <= 45.0)].mean(-1)
    ratio_ep = (beta / np.maximum(alpha, 1e-12)).mean(axis=1)
    
    # Cache ratios in metadata
    try:
        epochs.metadata = pd.DataFrame({"ba_ratio": ratio_ep}) if epochs.metadata is None else epochs.metadata.assign(ba_ratio=ratio_ep)
    except Exception:
        pass
    
    mask = ratio_ep < muscle_ratio_thr
    if not mask.any():
        print("🟥 [qc] no epochs passed beta/alpha ratio threshold")
        return mask  # Return all False - don't force keep bad epochs
    
    return mask


def _epoch_quality_scores(epochs: mne.Epochs) -> np.ndarray:
    """Return per-epoch quality scores (lower is better): beta/alpha ratio."""
    try:
        if epochs.metadata is not None and "ba_ratio" in epochs.metadata:
            return np.asarray(epochs.metadata["ba_ratio"].values, dtype=float)
    except Exception:
        pass
    
# === Metadata & File Processing ===
def load_metadata_harvard() -> dict:
    """Load and merge Harvard metadata CSVs into a lookup dict."""
    try:
        meta_files = glob.glob(os.path.join(METADATA_DIR, "*_eeg_metadata_*.csv"))
        dfs = []
        for f in meta_files:
            dfs.append(pd.read_csv(f))
        
        if not dfs:
            print("🟥 [metadata] No metadata CSVs found!")
            return {}
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Create lookup: {(sub_id, ses_id): {"age": age, "sex": sex}}
        # BidsFolder looks like 'sub-S0001...'
        # SessionID is int like 17
        
        lookup = {}
        for _, row in df.iterrows():
            sub = row['BidsFolder']
            ses = f"ses-{int(row['SessionID'])}" if pd.notna(row['SessionID']) else None
            
            if sub and ses:
                sex_str = str(row['SexDSC']).lower().strip()
                sex = 1 if sex_str == 'male' else (0 if sex_str == 'female' else -1)
                age = float(row['AgeAtVisit']) if pd.notna(row['AgeAtVisit']) else 0.0
                
                lookup[(sub, ses)] = {"age": age, "sex": sex}
                
        print(f"✅ [metadata] Loaded {len(lookup)} sessions")
        return lookup
    except Exception as e:
        print(f"🟥 [metadata] Failed to load: {e}")
        return {}



def _process_one_file(eeg_path: str, data_root: str, sfreq: int, epoch_len_s: float, split_name: str, ar_n_jobs: int, metadata: dict) -> dict:
    """Process a single EDF file and return tuples for batch saving."""
    mne.set_log_level('ERROR')  # Suppress filter messages in worker
    
    # Check filename pattern: sub-XXX_ses-YYY_...
    fn = os.path.basename(eeg_path)
    
    # Extract BIDS entities
    parts = fn.split('_')
    sub_id = next((p for p in parts if p.startswith('sub-')), None)
    ses_id = next((p for p in parts if p.startswith('ses-')), None)
    
    # Default label = 0 (Normal) for this dataset
    label = 0
    
    # Metadata lookup
    age, sex = 0, -1
    if sub_id and ses_id:
        meta = metadata.get((sub_id, ses_id))
        if meta:
            age = meta['age']
            sex = meta['sex']
    
    if sex == -1:
        # Fallback to EDF header if possible (unlikely reliable but try)
        # Or just skip if rigorous
        print(f"🟠 [skip] {fn}: sex unknown (not in metadata)")
        return {"file": fn, "epochs_saved": 0, "reason": "sex_unknown_in_metadata", "epoch_tuples": []}
        
    # Load and validate
    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
        
        # CROPPING STRATEGY: Limit to middle 30 mins if too long
        # This prevents 2GB+ files from exploding memory
        dur = raw.times[-1]
        MAX_DUR = 1800.0 # 30 mins
        
        if dur > MAX_DUR:
            mid = dur / 2
            tmin = mid - (MAX_DUR / 2)
            tmax = mid + (MAX_DUR / 2)
            raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
            # print(f"✂️ [crop] {fn}: {dur:.1f}s -> 1800s (middle)") 
             
    except Exception as e:
         return {"file": fn, "epochs_saved": 0, "reason": f"read_error: {e}", "epoch_tuples": []}

    # Clean
    try:
        clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
    except Exception as e:
        print(f"🟥 [skip] {eeg_path}: {e}")
        return {"file": fn, "epochs_saved": 0, "reason": f"cleanup_failed: {e}", "epoch_tuples": []}
    
    # Epoching + AR + QC
    try:
        ep = _make_fixed_length_epochs(clean, epoch_len_s)
    except Exception as e:
        print(f"🟠 [skip] {fn}: epoching failed (too short?): {e}")
        # Clean up memory just in case
        del clean
        del raw
        gc.collect()
        return {"file": fn, "epochs_saved": 0, "reason": "epoching_failed_too_short", "epoch_tuples": []}
    
    # Clean up raw to free memory before AutoReject
    del clean
    del raw
    gc.collect() 
    
    ep = _apply_autoreject(ep, ar_n_jobs=ar_n_jobs)
    
    if ep is None:
        return {"file": fn, "epochs_saved": 0, "reason": "autoreject_rejected_all", "epoch_tuples": []}
    
    mask = _qc_epoch_mask(ep)
    ep = ep[mask]
    
    if len(ep) == 0:
        return {"file": fn, "epochs_saved": 0, "reason": "no_epochs_after_qc", "epoch_tuples": []}
    
    # Cap at 20 epochs
    if len(ep) > 20:
        try:
            scores = _epoch_quality_scores(ep)
            keep_idx = np.argsort(scores)[:20]
            ep = ep[keep_idx]
        except:
            pass
    
    # Create epoch tuples (raw, g, a, ab, sample_id)
    edf_stem = os.path.splitext(fn)[0]
    epoch_tuples = []
    
    for i in range(len(ep)):
        epoch_data = ep.get_data()[i]
        info = mne.create_info(ep.info['ch_names'], sfreq=ep.info['sfreq'], ch_types='eeg')
        raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)
        
        _suppress(lambda: raw_epoch.set_montage(ep.get_montage(), verbose=False))
        
        epoch_id = f"{edf_stem}_epoch{i:04d}"
        raw_epoch.info["description"] = f"epoch_id={epoch_id}; abnormal={label}; sex={sex}; age={age}"
        
        subj = raw_epoch.info.get("subject_info", {}) or {}
        subj.update({"sex": int(sex), "his_id": str(epoch_id)})
        raw_epoch.info["subject_info"] = subj
        
        # Tuple: (raw, sex, age, label, sample_id)
        epoch_tuples.append((raw_epoch, sex, age, label, epoch_id))
    
    return {"file": fn, "epochs_saved": len(ep), "reason": "success", "epoch_tuples": epoch_tuples}



def _process_split(file_list: List[str], data_root: str, save_path: str, sfreq: int, epoch_len_s: float, split_name: str, n_jobs: int, ar_n_jobs: int, metadata: dict):
    """Process a list of EDF files and save to pickle files."""
    
    if not file_list:
        print(f"⚠️ [{split_name}] No files to process!")
        return []

    # Process in parallel
    # Force ar_n_jobs=1 internally to prevent explosion
    worker_ar_jobs = 1
    
    if tqdm_joblib is not None:
        with tqdm_joblib(tqdm(total=len(file_list), desc=f"[{split_name}] processing files", unit="file")):
            results = Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch='2*n_jobs')(
                delayed(_process_one_file)(eeg_path, data_root, sfreq, epoch_len_s, split_name, worker_ar_jobs, metadata)
                for eeg_path in file_list
            )
    else:
            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10, pre_dispatch='2*n_jobs')(
            delayed(_process_one_file)(eeg_path, data_root, sfreq, epoch_len_s, split_name, worker_ar_jobs, metadata)
            for eeg_path in file_list
        )
    
    # Filter None
    results = [r for r in results if r is not None]
    
    # Collect all epoch tuples
    all_epoch_tuples = []
    for result in results:
        if result["epoch_tuples"]:
            all_epoch_tuples.extend(result["epoch_tuples"])
            # Clear from result object to save memory
            result["epoch_tuples"] = None 
    
    # Explicit GC
    gc.collect()

    all_results = results


    # Save all epochs to single pickle file
    output_file = os.path.join(save_path, f"{split_name}_epochs.pkl")
    print(f"[{split_name}] Saving {len(all_epoch_tuples)} epochs to {output_file}")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_epoch_tuples, f, protocol=pickle.HIGHEST_PROTOCOL)


    # Report accumulated statistics
    insufficient_files = [r for r in all_results if r["epochs_saved"] < 20]
    
    if insufficient_files:
        print(f"\n🟠 [{split_name.upper()}] Files with <20 clean epochs:")
        for result in insufficient_files:
            print(f"  - {result['file']}: {result['epochs_saved']} epochs ({result['reason']})")
        print(f"Total files with <20 epochs: {len(insufficient_files)}/{len(all_results)}")
    else:
        print(f"\n✅ [{split_name.upper()}] All {len(all_results)} files produced ≥20 clean epochs!")
    
    return all_results


def load_data(data_path, save_path, sfreq=128, epoch_len_s: float = 10.0, n_jobs: int = 16, ar_n_jobs: Optional[int] = None):
    """Main entry point for Harvard EEG dataset processing."""
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Load Metadata
    metadata = load_metadata_harvard()
    
    # 2. Discover all EDF files
    print(f"🔍 Searching for EDF files in {data_path}...")
    all_files = []
    for root, _, files in os.walk(data_path):
        for f in files:
            if f.lower().endswith('.edf'):
                all_files.append(os.path.join(root, f))
    
    print(f"found {len(all_files)} files.")
    if not all_files:
        return

    # 3. Split Train/Eval (80/20) by Subject to avoid leakage
    # Group by subject
    sub_map = {}
    for f in all_files:
        fn = os.path.basename(f)
        parts = fn.split('_')
        sub = next((p for p in parts if p.startswith('sub-')), "unknown")
        sub_map.setdefault(sub, []).append(f)
        
    subjects = sorted(list(sub_map.keys()))
    import random
    random.seed(42)
    random.shuffle(subjects)
    
    split_idx = int(0.8 * len(subjects))
    train_subs = set(subjects[:split_idx])
    eval_subs = set(subjects[split_idx:])
    
    train_files = [f for s in train_subs for f in sub_map[s]]
    eval_files = [f for s in eval_subs for f in sub_map[s]]
    
    print(f"Suggest Split: {len(train_files)} train files, {len(eval_files)} eval files")
    
    # Auto-set AR jobs
    if ar_n_jobs is None:
        ar_n_jobs = 1 if (n_jobs > 1) else -1

    # 4. Process Splits
    train_results = _process_split(train_files, data_path, save_path, sfreq, epoch_len_s, "train", n_jobs, ar_n_jobs, metadata)
    eval_results = _process_split(eval_files, data_path, save_path, sfreq, epoch_len_s, "eval", n_jobs, ar_n_jobs, metadata)
    
    # Overall summary
    all_results = train_results + eval_results
    all_insufficient = [r for r in all_results if r["epochs_saved"] < 20]
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY:")
    print(f"Total files processed: {len(all_results)}")
    print(f"Files with ≥20 epochs: {len(all_results) - len(all_insufficient)}")
    print(f"Files with <20 epochs: {len(all_insufficient)}")
    
    if all_insufficient:
        print(f"\nALL FILES WITH <20 EPOCHS:")
        for result in all_insufficient:
            print(f"  - {result['file']}: {result['epochs_saved']} epochs ({result['reason']})")
    
    print("[save] Done. Data saved in train_epochs.pkl and eval_epochs.pkl")
    print("Each entry is a tuple: (raw, g, a, ab, sample_id)")
    print("  raw: mne.Raw object")
    print("  g: gender (0=female, 1=male)")  
    print("  a: age (years)")
    print("  ab: abnormal label (0=normal, 1=abnormal)")
    print("  sample_id: unique epoch identifier")
    print(f"{'='*60}")



if __name__ == "__main__":

    # Example usage
    data_path = "/rds/general/user/lrh24/ephemeral/harvard/EEG/bids_age_500"
    save_path = "/rds/general/user/lrh24/home/thesis/Datasets/harvard-eeg-clean"
    epoch_len_s = 10.0
    sfreq = 128
    n_jobs = 64 # Optimized for 256GB RAM (approx 16GB per worker peak)

    load_data(data_path, save_path, sfreq, epoch_len_s, n_jobs=n_jobs)
    print("[done] Data cleaning pipeline finished.")
