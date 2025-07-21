import mne
import os
from mne_bids import BIDSPath, read_raw_bids, get_entities_from_fname
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

import re

# ---------------------------------------------------------------------
# Harvard‑EEG abnormality definition (same columns used at sampling time)
# ---------------------------------------------------------------------
ABNORMAL_COLS = [
    "spikes", "lpd", "gpd", "lrda", "grda", "bs",
    "foc slowing", "gen slowing", "bipd", "status"
]
NORM_TOKENS = {"none", "normal", "", "nan", None}



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
            age = 0
            abn = 1 if path == "abnormal" else 0  # Abnormal
            t_data.append((raw, sex_code, abn))

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

    # ------------------------------------------------------------------
    # 🔍  Build fallback abnormal-session lookup from original CSVs
    #      (in case HEEDB_meta lacks the abnormality columns)
    # ------------------------------------------------------------------
    metadata_root = data_path.parent.parent / "metadata"  # ../.. / metadata
    abnormal_lookup = set()
    try:
        import pandas as _pd
        csvs_meta     = list(metadata_root.glob("*eeg_metadata_*.csv"))
        csvs_reports  = list(metadata_root.glob("*_EEG__reports_findings.csv"))
        if csvs_meta and csvs_reports:
            _read_csv_opts = {"dtype": str, "low_memory": False}
            meta_df    = _pd.concat([_pd.read_csv(fp, **_read_csv_opts) for fp in csvs_meta], ignore_index=True)
            reports_df = _pd.concat([_pd.read_csv(fp, **_read_csv_opts) for fp in csvs_reports], ignore_index=True)

            df = _pd.merge(meta_df, reports_df,
                           on=["SiteID", "BDSPPatientID", "SessionID"], how="inner")

            def _row_is_abnormal(row):
                vals = [row.get(col) for col in ABNORMAL_COLS]
                vals = ["none" if _pd.isna(v) else str(v).strip().lower() for v in vals]
                has_abn = any(v not in NORM_TOKENS for v in vals)
                seizure_txt = str(row.get("seizure", ""))
                if _pd.isna(seizure_txt):
                    seizure_txt = ""
                return has_abn and ("seizure" not in seizure_txt.lower())

            abn_rows = df[df.apply(_row_is_abnormal, axis=1)]
            for _, r in abn_rows.iterrows():
                subj = str(r["BidsFolder"]).replace("sub-", "") if not _pd.isna(r.get("BidsFolder")) else f"{r['SiteID']}{int(r['BDSPPatientID'])}"
                sess = str(int(r["SessionID"]))
                abnormal_lookup.add((subj, sess))
            print(f"🔎  Built abnormal lookup with {len(abnormal_lookup)} sessions.")
        else:
            print("⚠️  Could not find metadata CSVs – fallback abnormal lookup disabled.")
    except Exception as e:
        print(f"⚠️  Failed to build abnormal lookup: {e}")

    # Example filename: sub-XXXX_ses-YY_task-cEEG_eeg.vhdr
    subject_sessions = []  # tuples of (subject, session, task, vhdr_path)
    for subj_dir in subject_dirs:
        for ses_dir in subj_dir.glob("ses-*"):
            eeg_dir = ses_dir / "eeg"
            vhdr_files = list(eeg_dir.glob("*_eeg.vhdr"))
            if not vhdr_files:
                continue  # skip sessions with no EEG header

            for vhdr_path in vhdr_files:  # include *every* task file
                m = re.match(
                    r"sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>[^_]+)_eeg\.vhdr",
                    vhdr_path.name,
                )
                if not m:
                    print(f"⚠️  Could not parse entities from {vhdr_path.name}; skipping")
                    continue

                subject, session, task = m.group("sub", "ses", "task")
                subject_sessions.append((subject, session, task, vhdr_path))
            #print(f"Found subject-session pair: {subject}, {session}")

    # Split subject-session pairs into train/eval
    train_pairs, eval_pairs = train_test_split(subject_sessions, test_size=eval_split, random_state=42, shuffle=True)

    def process(pairs):
        result = []
        for i, (subject, session, task, vhdr_path) in enumerate(pairs):
            try:
                # Build BIDSPath directly from filename entities
                ent = get_entities_from_fname(str(vhdr_path))
                bids_path = BIDSPath(**ent, extension='.vhdr', root=data_path, datatype='eeg')
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    #warnings.filterwarnings("ignore", message="The search_str was .*events.tsv")
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

                # ----- Primary: infer from sidecar metadata -----
                def _meta_val(column: str):
                    return meta.get(column, meta.get(f"{column}_rep", ""))

                abnormal_hit = any(
                    str(_meta_val(col)).strip().lower() not in NORM_TOKENS
                    for col in ABNORMAL_COLS
                )
                seizure_txt = str(meta.get("seizure", meta.get("seizure_rep", ""))).strip().lower()
                if "seizure" in seizure_txt:
                    abnormal_hit = False

                # ----- Fallback: lookup table from CSVs -----
                if not abnormal_hit and (subject, str(int(session))) in abnormal_lookup:
                    abnormal_hit = True

                abn = int(abnormal_hit)
                #print(subject, session, "→ abnormal =", abn)
                # if i < 3:
                #     print(subject, session, "→ abnormal =", abn)
                
                #print(i,(raw, sex_code, age, abn))
                result.append((raw, sex_code, age, abn))

            except Exception as e:
                print(f"⚠️ Skipping {subject}, {session}: {e}")
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