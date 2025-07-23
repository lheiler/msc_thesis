from datetime import datetime, timezone
import shutil
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids
import mne
import numpy as np
import json
from pathlib import Path



import pandas as pd


def _safe_int(x):
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _read_concat(csv_dir: Path, glob_pat: str, **kwargs) -> pd.DataFrame:
    """Concatenate all CSVs that match `glob_pat` inside `csv_dir`"""
    frames = []
    for fp in csv_dir.glob(glob_pat):
        df = pd.read_csv(fp, **kwargs)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files match {glob_pat} in {csv_dir}")
    return pd.concat(frames, ignore_index=True)


def build_meta_lookup(csv_dir: Path):
    """
    Merge all per-site CSVs and return
    (subject_label, session_label) → metadata_dict
    where both labels match the on-disk BIDS folder names.
    """
    rep  = _read_concat(csv_dir, "*_EEG__reports_findings.csv")
    sess = _read_concat(csv_dir, "*_eeg_metadata_*.csv")
    med  = pd.read_csv(csv_dir / "HEEDB_Medication_ATC.csv")
    icd  = pd.read_csv(csv_dir / "HEEDB_ICD10_for_Neurology.csv")
    pat  = pd.read_csv(csv_dir / "HEEDB_patients.csv")

    # Ensure consistent dtypes
    for df in (rep, sess, med, icd, pat):
        df["BDSPPatientID"] = df["BDSPPatientID"].astype(str)

    # ---------- Merge (session table is the spine) ---------------------------
    meta = (
        sess
        .merge(rep, on=["SiteID", "BDSPPatientID", "SessionID"],
               how="left", suffixes=("", "_rep"))
        .merge(med, on=["SiteID", "BDSPPatientID"],
               how="left", suffixes=("", "_med"))
        .merge(icd, on=["SiteID", "BDSPPatientID"],
               how="left", suffixes=("", "_icd"))
        .merge(pat, on=["SiteID", "BDSPPatientID"],
               how="left", suffixes=("", "_pat"))
    )

    # ---------- Build lookup keyed EXACTLY as BIDS labels --------------------
    lookup = {}
    for _, row in meta.iterrows():
        # subject: prefer the explicit BidsFolder (e.g. "sub-S0001115722397")
        if "BidsFolder" in row and pd.notna(row.BidsFolder):
            subj = row.BidsFolder.replace("sub-", "")
        else:  # fall back to SiteID + BDSPPatientID
            subj = f"{row.SiteID}{_safe_int(row.BDSPPatientID)}"

        # session: cast to int to drop any leading zeros, then back to str
        sess_id = str(int(row.SessionID))

        mdict = (
            row.drop(["SiteID", "BDSPPatientID", "SessionID", "BidsFolder"], errors="ignore")
               .where(pd.notna(row))
               .to_dict()
        )
        lookup[(subj, sess_id)] = mdict

    print(f"🔎  Built metadata lookup for {len(lookup)} sessions")
    return lookup


def write_meta_to_sidecar(bids_root: Path, bids_path, meta_dict):
    """Same as before – unchanged."""
    sidecar = bids_path.copy().update(root=bids_root,
                                  suffix="eeg", extension=".json").fpath
    data = {}
    if sidecar.exists():
        data = json.loads(sidecar.read_text())
    data["HEEDB_meta"] = meta_dict
    sidecar.write_text(json.dumps(data, indent=2))




# === Setup ===
bids_root_in = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/bids_2000_normal_abnormal")
bids_root_out = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/bids_2000_normal_abnormal_clean")


# IMPORTANT: All the metadata CSVs must be in the same directory
csv_dir = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata") 

bids_root_out.mkdir(parents=True, exist_ok=True)

# === Define your 19-channel montage (10–20 subset) ===
standard_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]

duration = 60.0  # seconds

# All recordings will be assigned this same (arbitrary but valid) measurement date.
UNIFORM_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)




META_LOOKUP = build_meta_lookup(csv_dir)

# === Loop over EEG files ===
subject_dirs = sorted([p for p in bids_root_in.glob("sub-*") if p.is_dir()])

for subj_dir in subject_dirs:
    edf_files = list(subj_dir.rglob("*_eeg.edf"))
    
    for edf_file in edf_files:
        try:
            parts = edf_file.relative_to(bids_root_in).parts
            subject = [p for p in parts if p.startswith("sub-")][0].replace("sub-", "")
            session = [p for p in parts if p.startswith("ses-")][0].replace("ses-", "")
            task = edf_file.name.split("_task-")[1].split("_")[0]

            bids_path = BIDSPath(
                subject=subject,
                session=session,
                task=task,
                datatype="eeg",
                root=bids_root_in
            )

            # ------------------------------------------------------------
            # Skip this file if we have already written a cleaned version
            # (BrainVision header file *.vhdr inside the cleaned BIDS root)
            out_path = bids_path.copy().update(root=bids_root_out)
            out_vhdr = out_path.copy().update(suffix="eeg", extension=".vhdr").fpath
            if out_vhdr.exists():
                print(f"⏩  Skipping {bids_path.basename}: cleaned file already exists.")
                continue
            # ------------------------------------------------------------

            print(f"Processing {bids_path.basename}...")

            # === Load and clean data ===
            try:
                raw = read_raw_bids(bids_path, verbose=False)
            except ValueError as e:
                # Some sidecar JSON files contain a non-ISO timestamp like
                # "YYYY-mm-dd HH:MM:SS.xxxxxx" (space instead of "T"), which MNE-BIDS
                # fails to parse. Fall back to reading the EDF directly in that case.
                
                    # Ensure EEG reference is set like in the normal branch
                    print(str(e)+ "you fucked up")
            raw.set_meas_date(UNIFORM_DATE)
            raw.load_data()
            
            raw.set_eeg_reference('average', projection=False)

            # ✅ Pick only standard 19 channels (skip missing ones)
            available = [ch for ch in standard_19 if ch in raw.ch_names]
            if len(available) < 19:
                print(f"⚠️ Too few matching channels in {edf_file.name}, skipping")
                print(f"Available channels: {available}")
                print(f"all channels: {raw.ch_names}")
                
                continue
            raw.pick(available)
            
            raw.set_montage("standard_1020", match_case=False)

            # ✅ Resample to 128 Hz
            raw.resample(128, npad="auto")
            

            # ✅ Bandpass filter 1–40 Hz
            raw.filter(1., 40., fir_design='firwin', verbose=False)

            # ✅ Mark flat channels as bad
            flat_thresh = 1e-6  # µV
            flat_channels = [ch for ch in raw.ch_names
                             if np.std(raw.get_data(picks=ch)) < flat_thresh]
            raw.info['bads'] = flat_channels
            
            
            sfreq = raw.info['sfreq']  # should now be exactly 128
            n_samples = raw.n_times     # total number of samples

            target_len = int(duration * sfreq)  # 7680
            if n_samples >= target_len:
                start_sample = (n_samples - target_len) // 2
                stop_sample = start_sample + target_len
                raw.crop(tmin=start_sample / sfreq, tmax=(stop_sample / sfreq)- (1.0/sfreq))
            else:
                print(f"⚠️ Too short: only {n_samples} samples")
                continue
            
            # === Write cleaned version ===
            write_raw_bids(
                raw,
                bids_path=out_path,
                overwrite=True,
                allow_preload=True,
                format="BrainVision",
            )
            session_raw = session
            session_key = str(int(session_raw))
            meta = META_LOOKUP.get((subject, session), {})
            if meta:
                write_meta_to_sidecar(bids_root_out, out_path, meta)
            else:
                print(f"⚠️  No metadata found for ({subject}, {session})")


        except Exception as e:
            print(f"⚠️ Skipping {edf_file.name}: {e}")

print("✅ Done: Cleaned EEG saved to bids_root_small_clean/")
