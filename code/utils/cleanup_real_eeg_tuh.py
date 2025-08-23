"""
Cleaned EEG preprocessing pipeline for TUH dataset.
Refactored for conciseness while preserving all logic.
"""

import mne
import os
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne.time_frequency import psd_array_welch
from typing import List, Iterable, Optional
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from collections import Counter

# === Constants ===
EDGE_PAD_S: float = 0.5
HP_HZ: float = 0.5
LP_HZ: float = 45.0
ICA_N_COMPONENTS: float = 0.99

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
        raw_hp = raw.copy().filter(1.0, None, verbose=False)
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
            raw_clean = ica.apply(raw.copy(), verbose=False)
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
    
    # Fallback: compute from scratch
    X = epochs.get_data(copy=True)
    sf = epochs.info['sfreq']
    nper = int(max(2 * sf, 1))
    nover = int(max(1 * sf, 0))
    
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    
    alpha = psd[..., (freqs >= 8.0) & (freqs <= 12.0)].mean(-1)
    beta = psd[..., (freqs >= 20.0) & (freqs <= 45.0)].mean(-1)
    ratio = beta / np.maximum(alpha, 1e-12)
    
    return ratio.mean(axis=1)


# === File Processing ===
def _process_one_file(eeg_path: str, data_root: str, save_path_split: str, sfreq: int, epoch_len_s: float, split_name: str, ar_n_jobs: int) -> dict:
    """Process a single EDF file through the complete pipeline."""
    mne.set_log_level('ERROR')  # Suppress filter messages in worker
    
    # Extract labels from path
    root = os.path.dirname(eeg_path)
    fn = os.path.basename(eeg_path)
    rel = os.path.relpath(root, data_root)
    parts = [p.lower() for p in rel.split(os.sep) if len(p)]
    
    if 'abnormal' in parts:
        label = 1
    elif 'normal' in parts:
        label = 0
    else:
        print(f"🛑 [skip] {eeg_path}: unknown label (expected 'abnormal' or 'normal' in path)")
        return {"file": fn, "epochs_saved": 0, "reason": "unknown_label"}
    
    # Load and validate
    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    
    try:
        sex = int(raw.info['subject_info']['sex']) - 1
        if sex < 0:
            raise ValueError("Invalid sex")
    except Exception:
        print("🟥 [labels] sex unknown (0) -> discarding this recording")
        return {"file": fn, "epochs_saved": 0, "reason": "sex_unknown"}
    
    # Clean
    try:
        clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
    except Exception as e:
        print(f"🟥 [skip] {eeg_path}: {e}")
        return {"file": fn, "epochs_saved": 0, "reason": f"cleanup_failed: {e}"}
    
    # Epoching + AR + QC
    ep = _make_fixed_length_epochs(clean, epoch_len_s)
    n0 = len(ep)
    
    ep = _apply_autoreject(ep, ar_n_jobs=ar_n_jobs)
    
    # Check if AutoReject returned None (all epochs rejected)
    if ep is None:
        print(f"🟥 [summary] AutoReject rejected all epochs; skipping")
        return {"file": fn, "epochs_saved": 0, "reason": "autoreject_rejected_all"}
    
    mask = _qc_epoch_mask(ep)
    n_pass = int(mask.sum())
    ep = ep[mask]
    
    if len(ep) == 0:
        print(f"🟥 [summary] no {split_name} epochs kept; skipping")
        return {"file": fn, "epochs_saved": 0, "reason": "no_epochs_after_qc"}
    
    # Cap at 20 epochs, prefer best quality
    if len(ep) > 20:
        try:
            scores = _epoch_quality_scores(ep)
            keep_idx = np.argsort(scores)[:20]
            ep = ep[keep_idx]
            
            # Verify quality of kept epochs
            kept_scores = scores[keep_idx]
            if np.median(kept_scores) > 2.0:  # Same as muscle_ratio_thr
                print(f"🟠 [warning] Poor epoch quality: median beta/alpha = {np.median(kept_scores):.2f}")
        except Exception as e:
            print(f"🟥 [skip] Quality scoring failed: {e}; discarding sample")
            return {"file": fn, "epochs_saved": 0, "reason": "quality_scoring_failed"}
    else:
        print(f"🟠 [summary] only {len(ep)} epochs kept (< 20)")
        
        # Check quality when <20 epochs
        try:
            scores = _epoch_quality_scores(ep)
            if np.median(scores) > 2.0:
                print(f"🟠 [warning] Poor epoch quality: median beta/alpha = {np.median(scores):.2f}")
        except Exception:
            pass
    
    # Save epochs
    edf_stem = os.path.splitext(fn)[0]
    for i in range(len(ep)):
        epoch_data = ep.get_data()[i]
        info = mne.create_info(ep.info['ch_names'], sfreq=ep.info['sfreq'], ch_types='eeg')
        raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)
        
        _suppress(lambda: raw_epoch.set_montage(ep.get_montage(), verbose=False))
        
        epoch_id = f"{edf_stem}_epoch{i:04d}"
        raw_epoch.info["description"] = f"epoch_id={epoch_id}; abnormal={label}; sex={sex}"
        
        subj = raw_epoch.info.get("subject_info", {}) or {}
        subj.update({"sex": int(sex), "his_id": str(epoch_id)})
        raw_epoch.info["subject_info"] = subj
        
        out_path = os.path.join(save_path_split, f"{epoch_id}-raw.fif")
        raw_epoch.save(out_path, overwrite=True, verbose=False)
    
    return {"file": fn, "epochs_saved": len(ep), "reason": "success"}


def _process_split(data_path: str, save_path_split: str, sfreq: int, epoch_len_s: float, split_name: str, n_jobs: int = 1, ar_n_jobs: Optional[int] = None):
    """Process all EDF files in a split."""
    os.makedirs(save_path_split, exist_ok=True)
    
    # Discover files
    edf_files = []
    for root, _, files in os.walk(data_path):
        edf_files.extend(os.path.join(root, fn) for fn in files if fn.lower().endswith('.edf'))
    
    # Set AutoReject parallelism
    if ar_n_jobs is None:
        ar_n_jobs = 1 if (n_jobs is not None and int(n_jobs) != 1) else -1
    
    # Process in parallel
    if tqdm_joblib is not None:
        with tqdm_joblib(tqdm(total=len(edf_files), desc=f"[{split_name}] processing files", unit="file")):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_file)(eeg_path, data_path, save_path_split, sfreq, epoch_len_s, split_name, ar_n_jobs)
                for eeg_path in edf_files
            )
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_process_one_file)(eeg_path, data_path, save_path_split, sfreq, epoch_len_s, split_name, ar_n_jobs)
            for eeg_path in edf_files
        )
    
    # Filter and report files with <20 epochs
    results = [r for r in results if r is not None]  # Filter None results
    insufficient_files = [r for r in results if r["epochs_saved"] < 20]
    
    if insufficient_files:
        print(f"\n🟠 [{split_name.upper()}] Files with <20 clean epochs:")
        for result in insufficient_files:
            print(f"  - {result['file']}: {result['epochs_saved']} epochs ({result['reason']})")
        print(f"Total files with <20 epochs: {len(insufficient_files)}/{len(results)}")
    else:
        print(f"\n✅ [{split_name.upper()}] All {len(results)} files produced ≥20 clean epochs!")
    
    return results


def load_data(data_path_train, data_path_eval, save_path, sfreq=128, epoch_len_s: float = 10.0, n_jobs: int = 16, ar_n_jobs: Optional[int] = None):
    """Main entry point for TUH EEG dataset processing."""
    # Clean paths
    data_path_train = data_path_train.rstrip("/")
    data_path_eval = data_path_eval.rstrip("/")
    
    save_path_train = os.path.join(save_path, "train")
    save_path_eval = os.path.join(save_path, "eval")
    os.makedirs(save_path, exist_ok=True)
    
    # Process splits
    train_results = _process_split(data_path_train, save_path_train, sfreq, epoch_len_s, split_name="training", n_jobs=n_jobs, ar_n_jobs=ar_n_jobs)
    eval_results = _process_split(data_path_eval, save_path_eval, sfreq, epoch_len_s, split_name="evaluation", n_jobs=n_jobs, ar_n_jobs=ar_n_jobs)
    
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
    
    print("[save] Done. Labels embedded in each epoch's Raw.info (description & subject_info). No CSV written.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage
    data_path_train = "/rds/general/user/lrh24/ephemeral/edf/train"
    data_path_eval = "/rds/general/user/lrh24/ephemeral/edf/eval"
    save_path = "/rds/general/user/lrh24/ephemeral/tuh-eeg-ab-clean"
    epoch_len_s = 10.0
    sfreq = 128
    
    load_data(data_path_train, data_path_eval, save_path, sfreq, epoch_len_s)
    print("[done] Data cleaning pipeline finished.")
