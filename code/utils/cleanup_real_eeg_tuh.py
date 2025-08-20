import mne
import os 
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne.time_frequency import psd_array_welch


os.environ["JOBLIB_VERBOSE"] = "0" # Suppress joblib warnings

# Lightweight logger for pipeline progress
def _fmt_dur(sec: float) -> str:
    try:
        return f"{sec:.2f}s"
    except Exception:
        return str(sec)



# Canonical 10-20 EEG channel order (19 channels)
CANONICAL_19 = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2'
]

def cleanup_real_eeg_tuh(raw: mne.io.BaseRaw, sfreq, montage='standard_1020', ref_channels='average', segment_length=60, mains_freq: int = 60, verbose: bool = True) -> mne.io.BaseRaw:
    # simple local logger controlled by `verbose`
    if verbose:
        def _v(msg: str):
            print(f"[clean] {msg}")
    else:
        def _v(msg: str):
            pass

    
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

        # Map legacy 10-20 labels to modern 10-10
        legacy = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        if name in legacy:
            name = legacy[name]
        
        return name

    def _detect_and_interpolate_bad_channels(r: mne.io.BaseRaw):
        picks_eeg = mne.pick_types(r.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
        if len(picks_eeg) == 0:
            return
        data = r.get_data(picks=picks_eeg)
        # Peak-to-peak amplitude and variance per channel
        ptp = np.ptp(data, axis=1)
        var = np.var(data, axis=1)
        # Robust z-score on log-variance
        logv = np.log(var + 1e-12)
        med = np.median(logv)
        mad = np.median(np.abs(logv - med)) + 1e-12
        z = 0.6745 * (logv - med) / mad
        flat_idx = np.where(ptp < 5e-6)[0]          # < 5 µV over entire record
        high_idx = np.where(ptp > 500e-6)[0]        # > 500 µV gross artifacts/saturation
        out_idx = np.where(np.abs(z) > 3.5)[0]      # variance outliers
        bad_idx = sorted(set(flat_idx.tolist() + high_idx.tolist() + out_idx.tolist()))
        bads = [r.ch_names[picks_eeg[i]] for i in bad_idx]
        if len(bads):
            r.info['bads'] = sorted(set(r.info.get('bads', []) + bads))
            # Ensure we have headshape/montage for EEG interpolation; if missing, attach a standard montage
            try:
                has_dig = r.info.get('dig', None) is not None
            except Exception:
                has_dig = False
            if not has_dig:
                try:
                    r.set_montage('standard_1020', verbose=False)
                except Exception:
                    _v("⚠️ALARM: standard_1020 montage not found")
            r.interpolate_bads(reset_bads=True, verbose=False)
            if len(bads):
                _v(f"⚠️ALARM: interpolated bad channels: {len(bads)} -> {bads}")

    def _trim_zero_edges(r: mne.io.BaseRaw, eps: float = 0.0, min_keep_sec: float = 1.0):
        """Crop away leading and trailing samples that are exactly zero across all EEG channels.
        `eps` kept at 0.0 to target true zeros from EDF padding. Requires at least `min_keep_sec` to remain.
        """
        picks_eeg = mne.pick_types(r.info, eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
        if len(picks_eeg) == 0:
            return
        data = r.get_data(picks=picks_eeg)
        zero_mask = np.all(np.abs(data) <= eps, axis=0)
        n = zero_mask.size
        sf = r.info['sfreq']
        # find first non-zero sample
        if zero_mask[0]:
            nz_front = np.argmax(~zero_mask)
            if nz_front == 0 and zero_mask[0]:
                # all zeros
                return
        else:
            nz_front = 0
        # find last non-zero sample
        if zero_mask[-1]:
            nz_back = n - 1 - np.argmax(~zero_mask[::-1])
        else:
            nz_back = n - 1
        tmin = nz_front / sf
        tmax = nz_back / sf
        planned = (tmin, tmax)
        if tmax - tmin >= max(min_keep_sec, 1.0 / sf):
            r.crop(tmin=tmin, tmax=tmax, include_tmax=False)
            _v(f"trimmed zero-padding edges to [{_fmt_dur(planned[0])}, {_fmt_dur(planned[1])}] (kept {_fmt_dur(tmax - tmin)})")
    
    def _annotate_artifacts(r: mne.io.BaseRaw):
        """Annotate EOG blinks, ECG, muscle bursts, photic, and HV/other technician-marked periods.
        Creates BAD_* annotations so they can be excluded during epoching.
        """
        sfreq = r.info['sfreq']
        new_anns = []
        # ensure any new annotations share the same orig_time to allow concatenation
        orig_time = getattr(r.annotations, 'orig_time', None)

        # --- EOG / blinks ---
        eog_ch = None
        for ch in ('EOG', 'VEOG', 'HEOG', 'ROC', 'LOC'):
            if ch in r.ch_names:
                eog_ch = ch
                break
        r_for_eog = r
        if eog_ch is None and 'Fp1' in r.ch_names and 'Fp2' in r.ch_names:
            r_for_eog = r.copy()
            r_for_eog.set_bipolar_reference('Fp1', 'Fp2', ch_name='EOG-bip', drop_refs=False, copy=False, verbose=False)
            eog_ch = 'EOG-bip'
        if eog_ch is not None:
            try:
                eog_events = mne.preprocessing.find_eog_events(r_for_eog, ch_name=eog_ch, verbose=False)[0]
                if len(eog_events):
                    on = eog_events[:, 0] / sfreq
                    new_anns.append(mne.Annotations(on - 0.25, [0.5] * len(on), ['BAD_eog'] * len(on), orig_time=orig_time))
            except Exception:
                pass

        # --- ECG / cardiac ---
        if 'EKG1' in r.ch_names:
            try:
                ecg_events = mne.preprocessing.find_ecg_events(r, ch_name='EKG1', verbose=False)[0]
                if len(ecg_events):
                    on = ecg_events[:, 0] / sfreq
                    new_anns.append(mne.Annotations(on - 0.15, [0.3] * len(on), ['BAD_ecg'] * len(on), orig_time=orig_time))
            except Exception:
                pass

        # --- Muscle bursts ---
        try:
            muscle_ann, _ = mne.preprocessing.annotate_muscle_zscore(r, ch_type='eeg', threshold=4.0, min_length_good=0.2)
            if len(muscle_ann):
                new_anns.append(mne.Annotations(muscle_ann.onset, muscle_ann.duration, muscle_ann.description, orig_time=orig_time))
        except Exception:
            pass

        # --- EMG channel driven muscle detection (if EMG present) ---
        if 'EMG' in r.ch_names:
            try:
                emg = r.copy().pick_channels(['EMG'])
                emg.filter(20.0, 100.0, verbose=False)
                x = emg.get_data()[0]
                w = int(max(0.2 * sfreq, 1))  # 200 ms RMS window
                rms = np.sqrt(np.convolve(x * x, np.ones(w) / w, mode='same'))
                med = np.median(rms)
                mad = np.median(np.abs(rms - med)) + 1e-12
                thr = med + 5.0 * mad
                above = rms > thr
                # Find contiguous regions above threshold
                trans = np.diff(np.concatenate(([0], above.astype(int), [0])))
                starts = np.where(trans == 1)[0]
                ends = np.where(trans == -1)[0]
                onsets = starts / sfreq
                durations = (ends - starts) / sfreq
                keep = durations >= 0.2
                if keep.any():
                    new_anns.append(mne.Annotations(onsets[keep], durations[keep], ['BAD_muscle_emg'] * int(keep.sum()), orig_time=orig_time))
            except Exception:
                pass

        # --- Photic stimulation periods ---
        if 'PHOTIC' in r.ch_names:
            try:
                ev = mne.find_events(r, stim_channel='PHOTIC', shortest_event=1, verbose=False)
                if len(ev):
                    on = ev[:, 0] / sfreq
                    new_anns.append(mne.Annotations(on, [2.0] * len(on), ['BAD_photic'] * len(on), orig_time=orig_time))
            except Exception:
                pass

        # --- Technician annotations: hyperventilation, movement, etc. ---
        try:
            if len(r.annotations):
                bad_keys = ('hyper', 'hv', 'photic', 'test', 'artifact', 'movement', 'talk')
                starts, durations, descs = [], [], []
                for i, desc in enumerate(r.annotations.description):
                    d = desc.lower()
                    if any(k in d for k in bad_keys):
                        starts.append(r.annotations.onset[i])
                        dur = float(r.annotations.duration[i])
                        durations.append(dur if dur > 0 else 1.0)
                        # normalize label prefix
                        if not d.startswith('bad_'):
                            descs.append('BAD_' + desc)
                        else:
                            descs.append(desc)
                if starts:
                    new_anns.append(mne.Annotations(starts, durations, descs, orig_time=orig_time))
        except Exception:
            pass

        # count labels to summarize
        def _count_descs(annotations):
            from collections import Counter
            if annotations is None:
                return Counter()
            return Counter([d for d in annotations.description])

        before_counts = _count_descs(r.annotations)
        # Merge accumulated annotations
        for ann in new_anns:
            r.set_annotations(r.annotations + ann)
        after_counts = _count_descs(r.annotations)
        # compute delta for BAD_* labels
        deltas = {k: after_counts[k] - before_counts.get(k, 0) for k in after_counts.keys()}
        added_bad = {k: v for k, v in deltas.items() if k.upper().startswith('BAD_') and v > 0}
        if added_bad:
            summary = ", ".join([f"{k}:{v}" for k, v in sorted(added_bad.items())])
            _v(f"annotated artifacts -> {summary}")
    
    def _harmonic_notch_list(sfreq: float, mains: int):
        """Return list of mains harmonics up to Nyquist."""
        if mains is None or mains <= 0:
            return []
        n = int((sfreq / 2) // mains)
        return [mains * k for k in range(1, n + 1)]

    def _conform_to_canonical(r: mne.io.BaseRaw, canonical: list[str]):
        """Ensure r has exactly the canonical EEG channels in the given order.
        Missing channels are added as zeros. Helper channels already dropped.
        """
        missing = [ch for ch in canonical if ch not in r.ch_names]
        if missing:
            info_add = mne.create_info(missing, r.info['sfreq'], ch_types=['eeg'] * len(missing))
            data_add = np.zeros((len(missing), r.n_times), dtype=float)
            raw_add = mne.io.RawArray(data_add, info_add, verbose=False)
            r.add_channels([raw_add], force_update_info=True)
        # pick and reorder to canonical
        r.pick(canonical, verbose=False)
        try:
            r.set_montage('standard_1020', verbose=False)
        except Exception:
            print("⚠️ALARM: standard_1020 montage not found")
        return r

    # --- initial snapshot ---
    try:
        _v(f"input: {len(raw.ch_names)} channels, sfreq={raw.info.get('sfreq', 'NA')}, duration={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")
    except Exception:
        _v("input snapshot unavailable")

    # raw.crop(tmin=(raw.times[-1] / 2), tmax=segment_length+(raw.times[-1] / 2))  # Crop to the 60s segment of the recording
    # Housekeeping
    raw.rename_channels({ch: rename_channel(ch) for ch in raw.ch_names})
    raw.drop_channels(['T1', 'T2', '26', '27', '28', '29', '30'], on_missing='ignore')

    rocloc = raw.ch_names.count('ROC') > 0 and raw.ch_names.count('LOC') > 0
    if rocloc:
        raw.set_channel_types({'ROC': 'eog', 'LOC': 'eog'})

    # Broad channel typing for polygraphic sensors
    ecg_chs = [ch for ch in raw.ch_names if ('ECG' in ch.upper()) or ('EKG' in ch.upper())]
    if ecg_chs:
        raw.set_channel_types({ch: 'ecg' for ch in ecg_chs})
    if 'EMG' in raw.ch_names:
        raw.set_channel_types({'EMG': 'emg'})
    phot_chs = [ch for ch in raw.ch_names if 'PHOT' in ch.upper()]
    if phot_chs:
        raw.set_channel_types({ch: 'stim' for ch in phot_chs})
    misc_map = {ch: 'misc' for ch in ('IBI', 'BURSTS', 'SUPPR') if ch in raw.ch_names}
    if misc_map:
        raw.set_channel_types(misc_map)
    _v(f"channel typing: eeg={mne.pick_types(raw.info, eeg=True).size}, eog={mne.pick_types(raw.info, eog=True).size}, ecg={mne.pick_types(raw.info, ecg=True).size}, emg={mne.pick_types(raw.info, emg=True).size}, stim={mne.pick_types(raw.info, stim=True).size}, misc={mne.pick_types(raw.info, misc=True).size}")

    _detect_and_interpolate_bad_channels(raw)
    _v(f"post-initial interpolation: bads={raw.info.get('bads', [])}")

    raw.set_montage(montage, on_missing='ignore', verbose=False)  # Set montage to standard 10-20 system
    _trim_zero_edges(raw)
    _v(f"after montage+trim: {len(raw.ch_names)} channels, duration={_fmt_dur(raw.times[-1] if len(raw.times) else 0.0)}")

    # 1) Early resample to 256 Hz for speed and uniformity
    _v(f"resample: {raw.info['sfreq']} -> 256 Hz")
    raw.resample(256, npad="auto")

    # 2) High-pass to remove drift, then notch at mains and first harmonic
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    _v("high-pass: 0.5 Hz")
    freqs = _harmonic_notch_list(raw.info['sfreq'], mains_freq)
    if len(freqs):
        raw.notch_filter(freqs=freqs, verbose=False)
    if len(freqs):
        _v(f"notch: {freqs} Hz")
    else:
        _v("notch: none (no mains within Nyquist)")

    # 3) ICA
    # Fit ICA on EEG channels only (EOG/ECG are used for detection, not fitting)
    picks_eeg_only = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    n_comp = min(20, len(picks_eeg_only)) if len(picks_eeg_only) else 0
    if n_comp == 0:
        _v("ICA: skipped (no EEG channels available)")
        raw_clean = raw.copy()
    else:
        _v(f"ICA: fitting (method=infomax, n_components={n_comp})")
        ica = mne.preprocessing.ICA(n_components=n_comp, method='infomax', random_state=97, verbose=False)
        raw_hp = raw.copy().filter(1.0, None, verbose=False)
        ica.fit(raw_hp, picks=picks_eeg_only, decim=3, verbose=False)
        _v("ICA: fitted")
        excluded_before = set(ica.exclude)
        if rocloc:
            eog_idx, _ = ica.find_bads_eog(raw, ch_name=['ROC', 'LOC'], verbose=False)
            ica.exclude += eog_idx
            _v(f"ICA: EOG components via ROC/LOC -> {list(eog_idx)}")
        else:
            eog_cands = [ch for ch in ['EOG', 'VEOG', 'HEOG', 'Fp1', 'Fp2'] if ch in raw.ch_names]
            if eog_cands:
                eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_cands, verbose=False)
                ica.exclude += eog_idx
                _v(f"ICA: EOG components via {eog_cands} -> {list(eog_idx)}")
        ecg_for_ica = [ch for ch in raw.ch_names if ('ECG' in ch.upper()) or ('EKG' in ch.upper())]
        if ecg_for_ica:
            ecg_idx, _ = ica.find_bads_ecg(raw, ch_name=ecg_for_ica[0], verbose=False)
            ica.exclude += ecg_idx
            _v(f"ICA: ECG components via {ecg_for_ica[0]} -> {list(ecg_idx)}")
        _v(f"ICA: total components marked for exclusion -> {sorted(set(ica.exclude))}")
        raw_clean = ica.apply(raw.copy(), verbose=False)
        _v("ICA: components applied to raw copy")

    # 4) Rereference and low-pass anti-EMG
    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    _v("re-reference: average")
    raw_clean.filter(l_freq=None, h_freq=45., verbose=False)
    _v("low-pass: 45 Hz (anti-EMG)")

    _detect_and_interpolate_bad_channels(raw_clean)
    _v(f"post-ICA interpolation: bads={raw_clean.info.get('bads', [])}")

    # 4b) Annotate artifacts and stimulation/hyperventilation periods
    _annotate_artifacts(raw_clean)
    try:
        from collections import Counter
        c = Counter([d for d in raw_clean.annotations.description if str(d).upper().startswith('BAD_')])
        if c:
            _v("final BAD annotations: " + ", ".join([f"{k}:{v}" for k, v in sorted(c.items())]))
    except Exception:
        pass

    # Now drop helper channels not needed downstream
    raw_clean.drop_channels(['ROC', 'LOC', 'EKG1', 'EOG', 'VEOG', 'HEOG'], on_missing='ignore')
    _v(f"drop helpers: now {len(raw_clean.ch_names)} channels")

    # 5) Final resample to target frequency
    _v(f"final resample: {raw_clean.info['sfreq']} -> {sfreq} Hz")
    raw_clean.resample(sfreq, npad="auto")

    # Mark edges to avoid filter/transient contamination
    dur = raw_clean.times[-1] if len(raw_clean.times) else 0.0
    if dur > 2.0:
        edge = 0.5
        orig_time = getattr(raw_clean.annotations, 'orig_time', None)
        edge_anns = mne.Annotations([0, dur - edge], [edge, edge], ['BAD_edge', 'BAD_edge'], orig_time=orig_time)
        raw_clean.set_annotations(raw_clean.annotations + edge_anns)
        _v("marked edges: 2 BAD_edge annotations (0.5s each)")

    # Clear measurement date to avoid invalid placeholders
    raw_clean.set_meas_date(None)
    _v(f"meas_date set: None")

    # Enforce canonical 19-channel order for downstream models
    _conform_to_canonical(raw_clean, CANONICAL_19)
    _v(f"canonicalized: {len(raw_clean.ch_names)} channels -> order: {CANONICAL_19}")

    # Interpolate any newly added zero-filled channels to avoid flat channels
    _detect_and_interpolate_bad_channels(raw_clean)
    _v(f"post-canonical interpolation: bads={raw_clean.info.get('bads', [])}")

    return raw_clean


def _make_fixed_length_epochs(raw: mne.io.BaseRaw, epoch_len: float):
    print(f"[epochs] fixed-length segmentation: duration={epoch_len}s, no overlap; reject_by_annotation=False (AutoReject handles artifacts)")
    return mne.make_fixed_length_epochs(
        raw,
        duration=epoch_len,
        overlap=0.0,
        preload=True,
        verbose=False,
        reject_by_annotation=False,
    )

def _apply_autoreject(epochs: mne.Epochs) -> mne.Epochs:
    """Learn data-driven per-channel peak-to-peak thresholds and interpolate or drop
    on an epoch-wise basis using AutoReject (local). Returns cleaned epochs.
    """
    ar = AutoReject(n_jobs=-1, random_state=97, verbose=False)
    print("[autoreject] fitting local thresholds and transforming epochs")
    epochs_ar = ar.fit_transform(epochs)
    print(f"[autoreject] done; n_epochs_in={len(epochs)}, n_epochs_out={len(epochs_ar)}")
    return epochs_ar

def _qc_epoch_mask(epochs: mne.Epochs, muscle_ratio_thr: float = 2.0) -> np.ndarray:
    """Return boolean mask of epochs passing QC using beta/alpha power ratio.
    Keep epochs where (20–45 Hz)/(8–12 Hz) averaged across channels < threshold.
    If none pass, keep the single best epoch.
    """
    X = epochs.get_data(copy=True)  # (n_ep, n_ch, n_t)
    sf = epochs.info['sfreq']
    nper = int(max(2 * sf, 1))
    nover = int(max(1 * sf, 0))
    psd, freqs = psd_array_welch(X, sf, fmin=1.0, fmax=45.0, n_per_seg=nper, n_overlap=nover, average='mean', verbose=False)
    a = (freqs >= 8.0) & (freqs <= 12.0)
    b = (freqs >= 20.0) & (freqs <= 45.0)
    alpha = psd[..., a].mean(-1)
    beta = psd[..., b].mean(-1)
    ratio = beta / np.maximum(alpha, 1e-12)
    ratio_ep = ratio.mean(axis=1)
    mask = ratio_ep < muscle_ratio_thr
    if not mask.any():
        best = int(np.argmin(ratio_ep))
        keep = np.zeros_like(mask)
        keep[best] = True
        print("[qc] no epochs passed; keeping the single best epoch by ratio")
        pass_rate = float(keep.sum()) / float(len(keep)) if len(keep) else 0.0
        print(f"[qc] beta/alpha ratio gate: thr={muscle_ratio_thr}, pass={keep.sum()}/{len(keep)} ({pass_rate*100:.1f}%)")
        return keep
    pass_rate = float(mask.sum()) / float(len(mask)) if len(mask) else 0.0
    print(f"[qc] beta/alpha ratio gate: thr={muscle_ratio_thr}, pass={mask.sum()}/{len(mask)} ({pass_rate*100:.1f}%)")
    return mask

def _standardize_epoch_data(X: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel z-score: zero-mean, unit-variance along time."""
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True) + 1e-8
    return (X - mu) / sd

def _select_k_indices(n: int, k: int):
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=k, dtype=int)


def load_data(data_path_train, data_path_eval, save_path, sfreq=128, epoch_len_s: float = 10.0):  # TUH EEG dataset with fixed-length epoching
    if data_path_train.endswith("/"): data_path_train = data_path_train[:-1]
    if data_path_eval.endswith("/"): data_path_eval = data_path_eval[:-1]

    save_path_train = os.path.join(save_path, "train")
    save_path_eval = os.path.join(save_path, "eval")

    os.makedirs(save_path_train, exist_ok=True)
    os.makedirs(save_path_eval, exist_ok=True)

    train_labels = []
    eval_labels = []

    for root, dirs, files in os.walk(data_path_train):
        for fn in files:
            if not fn.lower().endswith('.edf'):
                continue
            eeg_path = os.path.join(root, fn)
            rel = os.path.relpath(root, data_path_train)
            parts = [p.lower() for p in rel.split(os.sep) if len(p)]
            if 'abnormal' in parts:
                label = 1
            elif 'normal' in parts:
                label = 0
            else:
                label = None
            print("Cleaning training data from:", eeg_path)
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            # Extract sex from EDF header subject_info; MNE: 0=unknown, 1=male, 2=female
            sex = 0
            try:
                subj = raw.info.get('subject_info', {}) or {}
                if isinstance(subj, dict):
                    val = subj.get('sex', 0)
                else:
                    val = getattr(subj, 'sex', 0)
                sex = int(val) if val is not None else 0
            except Exception:
                sex = 0
            if sex == 0:
                print("[labels] sex unknown (0) -> discarding this recording")
                continue
            sex_bin = 0 if sex == 1 else 1
            print(f"[labels] abnormal={label}, sex_raw={sex} -> sex_bin={sex_bin}")

            clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
            print(f"[summary] cleaned training record: channels={len(clean.ch_names)}, duration={_fmt_dur(clean.times[-1] if len(clean.times) else 0.0)}")

            ep = _make_fixed_length_epochs(clean, epoch_len_s)
            n0 = len(ep)
            ep = _apply_autoreject(ep)
            mask = _qc_epoch_mask(ep)
            n_pass = int(mask.sum())
            ep = ep[mask]
            print(f"[summary] epochs train: before={n0}, after_qc={n_pass}, kept_pct={(100.0*n_pass/max(1,n0)):.1f}%")
            if len(ep) == 0:
                print("[summary] WARNING: no train epochs kept for this record; skipping")
                continue
            edf_stem = os.path.splitext(fn)[0]
            for i in range(len(ep)):
                epoch_data = ep.get_data()[i]
                info = mne.create_info(ep.info['ch_names'], sfreq=ep.info['sfreq'], ch_types='eeg')
                raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)
                try:
                    m = ep.get_montage()
                    if m is not None:
                        raw_epoch.set_montage(m, verbose=False)
                except Exception:
                    pass
                epoch_id = f"{edf_stem}_epoch{i:04d}"
                out_path = os.path.join(save_path_train, f"{epoch_id}-raw.fif")
                raw_epoch.save(out_path, overwrite=True, verbose=False)
                train_labels.append({'id': epoch_id, 'y_abnormal': label, 'y_sex': sex_bin})

    for root, dirs, files in os.walk(data_path_eval):
        for fn in files:
            if not fn.lower().endswith('.edf'):
                continue
            eeg_path = os.path.join(root, fn)
            rel = os.path.relpath(root, data_path_eval)
            parts = [p.lower() for p in rel.split(os.sep) if len(p)]
            if 'abnormal' in parts:
                label = 1
            elif 'normal' in parts:
                label = 0
            else:
                label = None
            print("Cleaning evaluation data from:", eeg_path)
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            sex = 0
            try:
                subj = raw.info.get('subject_info', {}) or {}
                if isinstance(subj, dict):
                    val = subj.get('sex', 0)
                else:
                    val = getattr(subj, 'sex', 0)
                sex = int(val) if val is not None else 0
            except Exception:
                sex = 0
            if sex == 0:
                print("[labels] sex unknown (0) -> discarding this recording")
                continue
            sex_bin = 0 if sex == 1 else 1
            print(f"[labels] abnormal={label}, sex_raw={sex} -> sex_bin={sex_bin}")

            clean = cleanup_real_eeg_tuh(raw, sfreq=sfreq)
            print(f"[summary] cleaned eval record: channels={len(clean.ch_names)}, duration={_fmt_dur(clean.times[-1] if len(clean.times) else 0.0)}")

            ep = _make_fixed_length_epochs(clean, epoch_len_s)
            n0 = len(ep)
            ep = _apply_autoreject(ep)
            mask = _qc_epoch_mask(ep)
            n_pass = int(mask.sum())
            ep = ep[mask]
            print(f"[summary] epochs eval: before={n0}, after_qc={n_pass}, kept_pct={(100.0*n_pass/max(1,n0)):.1f}%")
            if len(ep) == 0:
                print("[summary] WARNING: no eval epochs kept for this record; skipping")
                continue
            edf_stem = os.path.splitext(fn)[0]
            for i in range(len(ep)):
                epoch_data = ep.get_data()[i]
                info = mne.create_info(ep.info['ch_names'], sfreq=ep.info['sfreq'], ch_types='eeg')
                raw_epoch = mne.io.RawArray(epoch_data, info, verbose=False)
                try:
                    m = ep.get_montage()
                    if m is not None:
                        raw_epoch.set_montage(m, verbose=False)
                except Exception:
                    pass
                epoch_id = f"{edf_stem}_epoch{i:04d}"
                out_path = os.path.join(save_path_eval, f"{epoch_id}-raw.fif")
                raw_epoch.save(out_path, overwrite=True, verbose=False)
                eval_labels.append({'id': epoch_id, 'y_abnormal': label, 'y_sex': sex_bin})

    # Save labels CSVs
    if len(train_labels):
        pd.DataFrame(train_labels).to_csv(os.path.join(save_path_train, 'labels.csv'), index=False)
        print(f"[save] train: saved {len(train_labels)} epochs and labels.csv")
    if len(eval_labels):
        pd.DataFrame(eval_labels).to_csv(os.path.join(save_path_eval, 'labels.csv'), index=False)
        print(f"[save] eval: saved {len(eval_labels)} epochs and labels.csv")

    return



if __name__ == "__main__":
    # Example usage
    data_path_train = "/rds/general/user/lrh24/ephemeral/edf/train"
    data_path_eval = "/rds/general/user/lrh24/ephemeral/edf/eval"
    save_path = "/rds/general/user/lrh24/ephemeral/tuh-eeg-ab-clean"
    epoch_len_s = 10.0
    sfreq = 128
    load_data(data_path_train, data_path_eval, save_path, sfreq, epoch_len_s)
    
    print("[done] Data cleaning pipeline finished.")
