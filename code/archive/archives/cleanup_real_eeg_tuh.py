# === Channel occurrence summary (across all files) ===
# EEG FP1-REF:  2993
# EEG FP2-REF:  2993
# EEG F3-REF:   2993
# EEG F4-REF:   2993
# EEG C3-REF:   2993
# EEG C4-REF:   2993
# EEG P3-REF:   2993
# EEG P4-REF:   2993
# EEG O1-REF:   2993
# EEG O2-REF:   2993
# EEG F7-REF:   2993
# EEG F8-REF:   2993
# EEG T3-REF:   2993
# EEG T4-REF:   2993
# EEG T5-REF:   2993
# EEG T6-REF:   2993
# EEG A1-REF:   2993
# EEG A2-REF:   2993
# EEG FZ-REF:   2993
# EEG CZ-REF:   2993
# EEG PZ-REF:   2993
# IBI:          2993
# BURSTS:       2993
# SUPPR:        2993
# EEG EKG1-REF: 2990
# EEG T1-REF:   2990
# EEG T2-REF:   2990
# PHOTIC-REF:   2836
# EEG ROC-REF:  2786
# EEG LOC-REF:  2786
# EMG-REF:      1814
# EEG 26-REF:   1663
# EEG 27-REF:   1638
# EEG 28-REF:   1638
# EEG 29-REF:   1638
# EEG 30-REF:   1638
# EEG C3P-REF:   111
# EEG C4P-REF:   111
# EEG 31-REF:    111
# EEG 32-REF:    111
# EEG SP1-REF:   105
# EEG SP2-REF:   105
# EEG OZ-REF:      3
# ECG EKG-REF:     3
# PULSE RATE:      3
# EEG PG1-REF:     2
# EEG PG2-REF:     2

# --- Imports and Constants ---
import mne
import os
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne.time_frequency import psd_array_welch
from typing import List, Iterable
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  
import matplotlib.pyplot as plt

EDGE_PAD_S: float = 0.5
HP_HZ: float = 0.5
LP_HZ: float = 45.0
ICA_N_COMPONENTS: float = 0.99


import logging
logging.getLogger('mne').setLevel(logging.ERROR)
mne.set_log_level('ERROR')


os.environ["JOBLIB_VERBOSE"] = "0"

CANONICAL_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'
]

# Helper set for early drops and cardiac channel detection
EXTRA_NONCANON = {'T1','T2','26','27','28','29','30','31','32','A1','A2','C3P','C4P','OZ','SP1','SP2','PG1','PG2','IBI','BURSTS','SUPPR'}

def _is_cardiac(name: str) -> bool:
    u = name.upper()
    return ('ECG' in u) or ('EKG' in u) or ('PULSE' in u)

# --- Centralized helpers for repeated logic ---
def _count_types(raw: mne.io.BaseRaw) -> str:
    eeg = mne.pick_types(raw.info, eeg=True).size
    eog = mne.pick_types(raw.info, eog=True).size
    ecg = mne.pick_types(raw.info, ecg=True).size
    emg = mne.pick_types(raw.info, emg=True).size
    stim = mne.pick_types(raw.info, stim=True).size
    misc = mne.pick_types(raw.info, misc=True).size
    return f"eeg={eeg}, eog={eog}, ecg={ecg}, emg={emg}, stim={stim}, misc={misc}"

def _drop_helpers(raw: mne.io.BaseRaw, verbose=print) -> None:
    cardiac = [ch for ch in raw.ch_names if _is_cardiac(ch)]
    eog = [ch for ch in ['ROC','LOC','EOG','VEOG','HEOG'] if ch in raw.ch_names]
    if cardiac or eog:
        raw.drop_channels(eog + cardiac, on_missing='ignore')
        verbose(f"[drop helpers] {len(raw.ch_names)} ch")

def _mad_clip_inplace(raw_hp: mne.io.BaseRaw, picks: Iterable[int], limit: float = 20.0) -> None:
    X = raw_hp.get_data(picks=picks)
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + 1e-12
    lim = limit * mad
    Xc = np.clip(X, med - lim, med + lim)
    raw_hp._data[picks, :] = Xc

# --- Conciseness helpers ---
def _suppress(fn, *a, **k):
    try: return fn(*a, **k)
    except Exception: return None

def _safe_set_montage(inst, montage='standard_1020'):
    try: inst.set_montage(montage, verbose=False)
    except Exception: print("🛑[canonical] standard_1020 montage not found")

# Normalize event outputs (MNE version differences): return 1-D sample indices
def _event_samples(ev):
    if ev is None:
        return np.array([], dtype=int)
    # Some funcs return (events, scores); others just events
    if isinstance(ev, tuple):
        ev = ev[0]
    arr = np.asarray(ev)
    if arr.ndim == 0:
        return np.array([], dtype=int)
    if arr.ndim == 1:
        return arr.astype(int)
    if arr.ndim == 2:
        return arr[:, 0].astype(int)
    return np.array([], dtype=int)

# --- Utility Functions ---
def _fmt_dur(sec: float) -> str:
    """Format seconds with 2 decimals, fallback to str on error."""
    try: return f"{sec:.2f}s"
    except: return str(sec)

def rename_channel(name: str) -> str:
    """Canonicalize TUH channel names.
    - Strip "EEG ", remove "-REF"
    - Normalize case (Fp, Fz, Cz, Pz)
    - Map legacy T3/4/5/6 → T7/T8/P7/P8
    """
    name = name.removeprefix("EEG ").removesuffix("-REF").strip().upper()
    if name.startswith("FP"): name = name.replace("FP", "Fp")
    if name == "FZ": name = "Fz"
    if name == "CZ": name = "Cz"
    if name == "PZ": name = "Pz"
    legacy = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    return legacy.get(name, name)


def trim_zero_edges(raw: mne.io.BaseRaw, eps: float = 0.0, min_keep_sec: float = 1.0, verbose=print) -> None:
    """Trim leading/trailing regions where all EEG samples are near zero.
    Keeps at least ``min_keep_sec`` to avoid empty records.
    """
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    if not len(picks): return
    data = raw.get_data(picks=picks)
    zero_mask = np.all(np.abs(data) <= eps, axis=0)
    n, sf = zero_mask.size, raw.info['sfreq']
    nz_front = np.argmax(~zero_mask) if zero_mask[0] else 0
    if nz_front == 0 and zero_mask[0]: return
    nz_back = n-1-np.argmax(~zero_mask[::-1]) if zero_mask[-1] else n-1
    tmin, tmax = nz_front/sf, nz_back/sf
    if tmax-tmin >= max(min_keep_sec, 1.0/sf):
        raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        verbose(f"[trim] zero edges to [{_fmt_dur(tmin)}, {_fmt_dur(tmax)}]")


def conform_to_canonical(raw: mne.io.BaseRaw, canonical: List[str]) -> mne.io.BaseRaw:
    """Pick and order channels to the canonical set; raise if any are missing."""
    missing = [ch for ch in canonical if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"🛑 [canonical] missing canonical channels: {missing}")
    raw.pick(canonical, verbose=False)
    _safe_set_montage(raw, 'standard_1020')
    return raw



def annotate_artifacts(raw: mne.io.BaseRaw, verbose=print) -> None:
    """Annotate artifact segments as BAD_* without altering data.
    Adds: BAD_muscle (z-score), BAD_muscle_emg (from EMG), BAD_flatline, BAD_clipping,
    BAD_step, BAD_photic, and technician-derived BAD_* from existing annotations.
    """
    sfreq = raw.info['sfreq']; orig_time = getattr(raw.annotations, 'orig_time', None); new_anns = []
    # Muscle (EEG-wide)
    mus = _suppress(mne.preprocessing.annotate_muscle_zscore, raw, ch_type='eeg', threshold=4.0, min_length_good=0.2)
    if mus is not None and len(mus[0]): new_anns.append(mne.Annotations(mus[0].onset, mus[0].duration, mus[0].description, orig_time=orig_time))
    # EMG channel bursts
    if 'EMG' in raw.ch_names:
        def _emg_ann():
            emg = raw.copy(); emg.pick('EMG'); emg.filter(20.,100.,verbose=False)
            x = emg.get_data()[0]; w = int(max(0.2*sfreq,1)); rms = np.sqrt(np.convolve(x*x, np.ones(w)/w, mode='same'))
            med, mad = np.median(rms), np.median(np.abs(rms-np.median(rms)))+1e-12; thr = med+5*mad; above = rms>thr
            trans = np.diff(np.concatenate(([0], above.astype(int), [0]))); s,e = np.where(trans==1)[0], np.where(trans==-1)[0]
            on, du = s/sfreq, (e-s)/sfreq; keep = du>=0.2
            if keep.any(): return mne.Annotations(on[keep], du[keep], ['BAD_muscle_emg']*int(keep.sum()), orig_time=orig_time)
        ann = _suppress(_emg_ann)
        if ann is not None: new_anns.append(ann)

    # Photic
    if 'PHOTIC' in raw.ch_names:
        def _safe_find_events():
            events = mne.find_events(raw, stim_channel='PHOTIC', shortest_event=1, verbose=False)
            if events is not None and len(events) > 0:
                return events.astype(int)  # Ensure integer type
            return None
        ev = _suppress(_safe_find_events)
        smp = _event_samples(ev)
        if smp.size:
            on = smp / sfreq
            new_anns.append(mne.Annotations(on, [2.0]*len(on), ['BAD_photic']*len(on), orig_time=orig_time))
    # Technician-derived BAD_*
    if len(raw.annotations):
        #print(f"[annotate_artifacts] {len(raw.annotations)} annotations")
        bad_keys = ('hyper','hv','photic','test','artifact','movement','talk')
        sel = [i for i,d in enumerate(raw.annotations.description) if any(k in str(d).lower() for k in bad_keys)]
        if sel:
            starts = [raw.annotations.onset[i] for i in sel]
            durs = [float(raw.annotations.duration[i]) if float(raw.annotations.duration[i])>0 else 1.0 for i in sel]
            descs = ['BAD_'+raw.annotations.description[i] if not str(raw.annotations.description[i]).lower().startswith('bad_') else raw.annotations.description[i] for i in sel]
            new_anns.append(mne.Annotations(starts, durs, descs, orig_time=orig_time))
    # Merge and summarize
    for ann in new_anns: raw.set_annotations(raw.annotations + ann)
    from collections import Counter
    c = Counter([d for d in raw.annotations.description if str(d).upper().startswith('BAD_')])
    if c: verbose("[annots] " + ", ".join([f"{k}:{v}" for k,v in sorted(c.items())]))

# --- Main Cleaning Function ---
def cleanup_real_eeg_tuh(
    raw: mne.io.BaseRaw,
    sfreq: float,
    montage: str = 'standard_1020',
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """End-to-end cleaning for a TUH EEG recording.

    Steps:
    1) Rename/pick channels, set montage, trim zero edges
    2) Resample→256 Hz, band-pass
    3) ICA (EOG/ECG component removal when present)
    4) Average reference, annotate artifacts, drop helper channels
    5) Resample→``sfreq``, add short edge masks, set meas_date=None
    6) Pick canonical 19 channels
    Returns a cleaned ``Raw`` ready for epoching.
    """
    log = print if verbose else (lambda *a,**k: None)
    log(f"[input] {len(raw.ch_names)} ch, sfreq={raw.info.get('sfreq','NA')}, dur={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")
    raw.rename_channels({ch: rename_channel(ch) for ch in raw.ch_names})
    raw.drop_channels(list(EXTRA_NONCANON), on_missing='ignore')
    if raw.ch_names.count('ROC') and raw.ch_names.count('LOC'):
        raw.set_channel_types({'ROC':'eog','LOC':'eog'})
    m = {ch:'ecg' for ch in raw.ch_names if _is_cardiac(ch)}
    m.update({ch:'emg' for ch in raw.ch_names if 'EMG' in ch.upper()})
    m.update({ch:'stim' for ch in raw.ch_names if 'PHOT' in ch.upper()})
    m.update({ch:'misc' for ch in ('IBI','BURSTS','SUPPR') if ch in raw.ch_names})
    if m: raw.set_channel_types(m)
    log(f"[types] {_count_types(raw)}")
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    trim_zero_edges(raw, verbose=log)
    log(f"[mont+trim] {len(raw.ch_names)} ch, dur={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")
    log(f"[resample] {raw.info['sfreq']} -> 256 Hz")
    raw.resample(256, npad="auto")
    raw.filter(l_freq=HP_HZ, h_freq=LP_HZ, verbose=False); log(f"[bp] {HP_HZ}-{LP_HZ} Hz")
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    n_comp = ICA_N_COMPONENTS
    if not n_comp:
        log("[ICA] skipped (no EEG)"); raw_clean = raw.copy()
    else:
        log(f"[ICA] fitting (picard, n={n_comp}, decim=5)")
        ica = mne.preprocessing.ICA(n_components=n_comp, method='picard', random_state=97, verbose=False)
        raw_hp = raw.copy().filter(1.0, None, verbose=False)
        X = raw_hp.get_data(picks=picks_eeg)
        # Guard against near-zero-variance channels in picks (numerical stability)
        stds = X.std(axis=1); good = stds > 1e-12
        if not np.all(good):
            bad_names = [raw.ch_names[picks_eeg[i]] for i in np.where(~good)[0]]
            print(f"🟥 [ICA] excluding near-zero-variance channels: {bad_names}")
            picks_eeg = picks_eeg[good]; n_comp = min(n_comp, len(picks_eeg))
        if n_comp:
            # MAD clipping to reduce outlier impact before ICA
            _mad_clip_inplace(raw_hp, picks_eeg, limit=20.0)
            ica.fit(raw_hp, picks=picks_eeg, decim=5, verbose=False)
            if raw.ch_names.count('ROC') and raw.ch_names.count('LOC'):
                eog_idx, _ = ica.find_bads_eog(raw, ch_name=['ROC','LOC'], verbose=False)
                ica.exclude += eog_idx; log(f"[ICA] EOG ROC/LOC: {list(eog_idx)}")
            else:
                eog_cands = [ch for ch in ['EOG','VEOG','HEOG','Fp1','Fp2'] if ch in raw.ch_names]
                if eog_cands:
                    eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_cands, verbose=False)
                    ica.exclude += eog_idx; log(f"[ICA] EOG {eog_cands}: {list(eog_idx)}")
            ecg_cands = [ch for ch in raw.ch_names if ('ECG' in ch.upper()) or ('EKG' in ch.upper())]
            if ecg_cands:
                ecg_idx, _ = ica.find_bads_ecg(raw, ch_name=ecg_cands[0], verbose=False)
                ica.exclude += ecg_idx; log(f"[ICA] ECG {ecg_cands[0]}: {list(ecg_idx)}")
            log(f"[ICA] exclude: {sorted(set(ica.exclude))}")
            raw_clean = ica.apply(raw.copy(), verbose=False)
            log("[ICA] applied")
        else:
            raw_clean = raw.copy()
    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    log("[ref] average")
    annotate_artifacts(raw_clean, verbose=log)
    # Drop helper channels dynamically
    _drop_helpers(raw_clean, verbose=log)
    log(f"[final resample] {raw_clean.info['sfreq']} -> {sfreq} Hz")
    raw_clean.resample(sfreq, npad="auto")
    dur = raw_clean.times[-1] if len(raw_clean.times) else 0.0
    if dur > 2.0:
        edge = EDGE_PAD_S
        orig_time = getattr(raw_clean.annotations, 'orig_time', None)
        edge_anns = mne.Annotations([0, dur-edge], [edge, edge], ['BAD_edge','BAD_edge'], orig_time=orig_time)
        raw_clean.set_annotations(raw_clean.annotations + edge_anns)
        log(f"[edge] marked {EDGE_PAD_S}s edges")
    raw_clean.set_meas_date(None)
    log("[meas_date] None")
    # Pre-check for canonical channels before picking
    missing = [ch for ch in CANONICAL_19 if ch not in raw_clean.ch_names]
    if missing:
        print(f"🟥 [fail] missing canonical channels before pick: {missing}")
        raise ValueError("missing canonical channels before pick")
    conform_to_canonical(raw_clean, CANONICAL_19)
    # No bad-channel gating; proceed without checking raw_clean.info['bads']
    return raw_clean



# (Coverage debug removed by request)


def _make_fixed_length_epochs(raw: mne.io.BaseRaw, epoch_len: float) -> mne.Epochs:
    """Segment into fixed, non-overlapping epochs. Drops epochs overlapping BAD_* annotations."""
    
    return mne.make_fixed_length_epochs(
        raw,
        duration=epoch_len,
        overlap=0.0,
        preload=True,
        verbose=False,
        reject_by_annotation=True,
    )

def _apply_autoreject(epochs: mne.Epochs, ar_n_jobs: int = -1) -> mne.Epochs:
    """Learn data-driven per-channel peak-to-peak thresholds and interpolate or drop
    on an epoch-wise basis using AutoReject (local). Returns cleaned epochs.
    """
    n_ep = len(epochs)
    if n_ep < 2:
        print("🟥 [autoreject] skipping: insufficient epochs (n<2)")
        return epochs
    
    def _try_autoreject(ep_copy):
        ar = AutoReject(n_interpolate=[0], n_jobs=ar_n_jobs, random_state=97, verbose=False)
        return ar.fit_transform(ep_copy)
    
    try:
        # First attempt with original data
        epochs_ar = _try_autoreject(epochs.copy())
        if len(epochs_ar) == 0:
            print("🟠 [autoreject] rejected all epochs; trying with µV→V scaling")
            # Check signal magnitude and try scaling if needed
            data_range = epochs.get_data().max() - epochs.get_data().min()
            if data_range > 1e-3:  # Likely in µV (>1mV range)
                epochs_scaled = epochs.copy()
                epochs_scaled._data *= 1e-6  # µV to V
                epochs_ar = _try_autoreject(epochs_scaled)
                if len(epochs_ar) > 0:
                    print(f"[autoreject] µV scaling worked; n_epochs_out={len(epochs_ar)}")
                    return epochs_ar
                else:
                    print("🟥 [autoreject] scaling failed too; keeping original epochs")
                    return epochs
            else:
                print("🟥 [autoreject] all rejected but data seems properly scaled; keeping original")
                return epochs
        else:
            #print(f"[autoreject] done; n_epochs_in={len(epochs)}, n_epochs_out={len(epochs_ar)}")
            return epochs_ar
    except Exception as e:
        print(f"🟥 [autoreject] failed ({e}); keeping original epochs without AR")
        return epochs



def _qc_epoch_mask(epochs: mne.Epochs, muscle_ratio_thr: float = 2.0) -> np.ndarray:
    """Return boolean mask of epochs passing QC using beta/alpha power ratio.
    Keep epochs where (20–45 Hz)/(8–12 Hz) averaged across channels < threshold.
    If none pass, keep the single best epoch.
    """
    X = epochs.get_data(copy=True)  # (n_ep, n_ch, n_t)
    sf = epochs.info['sfreq']
    nper = int(max(1.25 * sf, 1))
    nover = int(max(0.5 * sf, 0))
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    a = (freqs >= 8.0) & (freqs <= 12.0)
    b = (freqs >= 20.0) & (freqs <= 45.0)
    alpha = psd[..., a].mean(-1)
    beta = psd[..., b].mean(-1)
    ratio = beta / np.maximum(alpha, 1e-12)
    ratio_ep = ratio.mean(axis=1)
    # Cache ratios in epochs.metadata to avoid recomputation later
    try:
        if epochs.metadata is None:
            epochs.metadata = pd.DataFrame({"ba_ratio": ratio_ep})
        else:
            epochs.metadata = epochs.metadata.assign(ba_ratio=ratio_ep)
    except Exception:
        pass
    mask = ratio_ep < muscle_ratio_thr
    if not mask.any():
        best = int(np.argmin(ratio_ep))
        keep = np.zeros_like(mask)
        keep[best] = True
        print("🟥 [qc] no epochs passed; keeping the single best epoch by ratio")
        pass_rate = float(keep.sum()) / float(len(keep)) if len(keep) else 0.0
        #print(f"[qc] beta/alpha ratio gate: thr={muscle_ratio_thr}, pass={keep.sum()}/{len(keep)} ({pass_rate*100:.1f}%)")
        return keep
    pass_rate = float(mask.sum()) / float(len(mask)) if len(mask) else 0.0
    #print(f"[qc] beta/alpha ratio gate: thr={muscle_ratio_thr}, pass={mask.sum()}/{len(mask)} ({pass_rate*100:.1f}%)")
    return mask

# --- Helper to score epochs by beta/alpha ratio, matching QC ---
def _epoch_quality_scores(epochs: mne.Epochs) -> np.ndarray:
    """Return a per-epoch quality score (lower is better): beta/alpha ratio averaged over channels.
    Mirrors the computation used in _qc_epoch_mask.
    """
    # Prefer cached ratios if available
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
    a = (freqs >= 8.0) & (freqs <= 12.0)
    b = (freqs >= 20.0) & (freqs <= 45.0)
    alpha = psd[..., a].mean(-1)
    beta = psd[..., b].mean(-1)
    ratio = beta / np.maximum(alpha, 1e-12)
    return ratio.mean(axis=1)






# --- New helpers to enable parallel per-file processing ---
def _process_one_file(eeg_path: str, data_root: str, save_path_split: str, sfreq: int, epoch_len_s: float, split_name: str, ar_n_jobs: int) -> None:
    import mne
    mne.set_log_level('ERROR')  # Suppress filter messages in worker
    root = os.path.dirname(eeg_path)
    fn = os.path.basename(eeg_path)
    rel = os.path.relpath(root, data_root)
    parts = [p.lower() for p in rel.split(os.sep) if len(p)]
    if 'abnormal' in parts: label = 1
    elif 'normal' in parts: label = 0
    else:
        print(f"🛑 [skip] {eeg_path}: unknown label (expected 'abnormal' or 'normal' in path)")
        return
    #print(f"[{split_name}] {eeg_path}")
    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
    # Sex gating from EDF header
    sex = -1
    try:
        sex = int(raw.info['subject_info']['sex']); sex -= 1
    except Exception:
        print("🟥 [labels] sex unknown (0) -> discarding this recording"); return
    if sex < 0:
        print("🟥 [labels] sex unknown (0) -> discarding this recording"); return

    #print(f"[labels] abnormal={label}, sex={sex}")
    # Clean
    try:
        clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
    except Exception as e:
        print(f"🟥 [skip] {eeg_path}: {e}")
        return

    # Epoching + AR + QC
    ep = _make_fixed_length_epochs(clean, epoch_len_s)
    n0 = len(ep)
    ep = _apply_autoreject(ep, ar_n_jobs=ar_n_jobs)
    mask = _qc_epoch_mask(ep)
    n_pass = int(mask.sum())
    ep = ep[mask]
    #print(f"[summary] epochs {split_name}: before={n0}, after_qc={n_pass}, kept_pct={(100.0*n_pass/max(1,n0)):.1f}%")
    if len(ep) == 0:
        print(f"🟥 [summary] no {split_name} epochs kept; skipping")
        return
    # Cap per-record saved epochs to at most 20, preferring best by beta/alpha
    if len(ep) > 20:
        try:
            scores = _epoch_quality_scores(ep)
            keep_idx = np.argsort(scores)[:20]
            ep = ep[keep_idx]
            #print(f"[summary] selecting top 20 epochs by beta/alpha (median={np.median(scores[keep_idx]):.3f})")
        except Exception as e:
            ep = ep[:20]
            print(f"🟠[summary] quality scoring failed ({e}); keeping first 20 epochs")
    else: print(f"🟠 [summary] less than 20 epochs kept")
    # Save epochs
    edf_stem = os.path.splitext(fn)[0]
    for i in range(len(ep)):
        epoch_data = ep.get_data()[i]
        info = mne.create_info(ep.info['ch_names'], sfreq=ep.info['sfreq'], ch_types='eeg')
        raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)
        _suppress(lambda: raw_epoch.set_montage(ep.get_montage(), verbose=False))
        epoch_id = f"{edf_stem}_epoch{i:04d}"
        # --- Embed labels/metadata into Raw.info ---
        raw_epoch.info["description"] = f"epoch_id={epoch_id}; abnormal={label}; sex={sex}"
        subj = raw_epoch.info.get("subject_info", {}) or {}
        try:
            subj["sex"] = int(sex)
        except Exception:
            pass
        subj["his_id"] = str(epoch_id)
        raw_epoch.info["subject_info"] = subj
        out_path = os.path.join(save_path_split, f"{epoch_id}-raw.fif")
        raw_epoch.save(out_path, overwrite=True, verbose=False)
    return


def _process_split(data_path: str, save_path_split: str, sfreq: int, epoch_len_s: float, split_name: str, n_jobs: int = 1, ar_n_jobs: int | None = None):
    os.makedirs(save_path_split, exist_ok=True)
    edf_files = []
    for root, _, files in os.walk(data_path):
        for fn in files:
            if fn.lower().endswith('.edf'):
                edf_files.append(os.path.join(root, fn))
    if ar_n_jobs is None: ar_n_jobs = 1 if (n_jobs is not None and int(n_jobs) != 1) else -1
    # Parallel execution with progress
    if tqdm_joblib is not None:
        with tqdm_joblib(tqdm(total=len(edf_files), desc=f"[{split_name}] processing files", unit="file")):
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_file)(eeg_path, data_path, save_path_split, sfreq, epoch_len_s, split_name, ar_n_jobs)
                for eeg_path in edf_files
            )
    else:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_process_one_file)(eeg_path, data_path, save_path_split, sfreq, epoch_len_s, split_name, ar_n_jobs)
            for eeg_path in edf_files
        )
    return


def load_data(data_path_train, data_path_eval, save_path, sfreq=128, epoch_len_s: float = 10.0, n_jobs: int = 16, ar_n_jobs: int | None = None):  # TUH EEG dataset with fixed-length epoching
    if data_path_train.endswith("/"): data_path_train = data_path_train[:-1]
    if data_path_eval.endswith("/"): data_path_eval = data_path_eval[:-1]
    save_path_train = os.path.join(save_path, "train"); save_path_eval = os.path.join(save_path, "eval")
    os.makedirs(save_path, exist_ok=True)

    _process_split(data_path_train, save_path_train, sfreq, epoch_len_s, split_name="training", n_jobs=n_jobs, ar_n_jobs=ar_n_jobs)
    _process_split(data_path_eval, save_path_eval, sfreq, epoch_len_s, split_name="evaluation", n_jobs=n_jobs, ar_n_jobs=ar_n_jobs)
    print("🟢 [save] Done. Labels embedded in each epoch's Raw.info (description & subject_info). No CSV written.")
    return





if __name__ == "__main__":
    # Example usage
    data_path_train = "/rds/general/user/lrh24/ephemeral/edf/train"
    data_path_eval = "/rds/general/user/lrh24/ephemeral/edf/eval"
    save_path = "/rds/general/user/lrh24/ephemeral/tuh-eeg-ab-clean"
    epoch_len_s = 10.0
    sfreq = 128
    load_data(data_path_train, data_path_eval, save_path, sfreq, epoch_len_s)
    
    print("🟢 [done] Data cleaning pipeline finished.")
