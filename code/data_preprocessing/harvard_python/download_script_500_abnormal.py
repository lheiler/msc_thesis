import pandas as pd
import os

# ------------------------------------------------------------
# Set the number of sessions you want for *each* class here.
# Simply change `n` to control how many normal and abnormal EEGs
# will be downloaded.
# ------------------------------------------------------------
n = 500  # <-- adjust this value as needed
n_normal = n
n_abnormal = n

desired_dir = "/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG/"

# Load EEG metadata
meta1 = pd.read_csv("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata/S0001_eeg_metadata_2025-05-06.csv")
meta2 = pd.read_csv("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata/S0002_eeg_metadata_2025-05-06.csv")

# Load EEG reports
reports1 = pd.read_csv("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata/S0001_EEG__reports_findings.csv")
reports2 = pd.read_csv("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata/S0002_EEG__reports_findings.csv")

# Combine metadata and reports
meta = pd.concat([meta1, meta2], ignore_index=True)
reports = pd.concat([reports1, reports2], ignore_index=True)

# Calculate duration in minutes
meta["DurationMinutes"] = meta["DurationInSeconds"] / 60

abnormal_cols = [
    "spikes", "lpd", "gpd", "lrda", "grda", "bs", "foc slowing", "gen slowing", "bipd", "status"
]

# Merge safely on 3 keys
df = pd.merge(meta, reports, on=["SiteID", "BDSPPatientID", "SessionID"], how="inner")

#print("Available columns from LLM-extracted reports:")
#print(df.columns.tolist())

# Define normal EEGs (no abnormalities reported)
normal_df = df[
    df[abnormal_cols].fillna("none").apply(lambda row: all(val in ["none", "normal", "", None] for val in row), axis=1)
]

# ----------------------------------------------------------------------------
# 1) Identify numeric age column automatically (or raise if not found)
# 2) Stratify data into age deciles and sample *n* sessions from each class while
#    keeping the age distribution matched between normal and abnormal groups.
# ----------------------------------------------------------------------------

abnormal_df = df[
    (df[abnormal_cols].fillna("none").apply(lambda row: any(val not in ["none", "normal", "", None] for val in row), axis=1)) &
    (~df["seizure"].fillna("").str.contains("seizure", case=False))
]

# Quick sanity-check for available size before proceeding
if len(normal_df) < n:
    raise ValueError(f"Not enough normal EEGs. Found {len(normal_df)}, need at least {n}.")
if len(abnormal_df) < n:
    raise ValueError(f"Not enough abnormal EEGs. Found {len(abnormal_df)}, need at least {n}.")

# -------------------------------------------------------------------------
# Detect an age column (numeric) – adjust manually if detection fails
# -------------------------------------------------------------------------

age_column_candidates = [c for c in df.columns if "age" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
if not age_column_candidates:
    raise ValueError("No numeric age column found – please specify your age column name manually in the script.")

age_col = age_column_candidates[0]

# Bin ages into deciles (duplicates="drop" avoids bin edges collision if not enough unique ages)
df["age_bin"] = pd.qcut(df[age_col], q=10, duplicates="drop")

# Add age_bin to original splits
normal_df = normal_df.join(df[["age_bin"]])
abnormal_df = abnormal_df.join(df[["age_bin"]])

# -------------------------------------------------------------------------
# Precise age-bin balancing: allocate exact target per bin so that the final
# dataset has *n* sessions per class AND an identical age-distribution.
# -------------------------------------------------------------------------

# Determine capacity (the max we can safely draw from each bin)
bin_values = df["age_bin"].dropna().unique()
capacity_per_bin = {
    bin_val: min(len(normal_df[normal_df["age_bin"] == bin_val]),
                 len(abnormal_df[abnormal_df["age_bin"] == bin_val]))
    for bin_val in bin_values
}

if sum(capacity_per_bin.values()) < n:
    raise ValueError(
        "Not enough age-balanced samples across bins to satisfy the requested 'n'. "
        "Consider lowering 'n' or reducing the number of bins.")

# ---------------- Adaptive allocation -------------------------
# 1) Start with equal base allocation per bin.
# 2) If a bin lacks enough capacity, allocate what it can and queue the
#    deficit. Any remaining deficit is redistributed to bins with spare
#    capacity until all `n` pairs are assigned.
# --------------------------------------------------------------

base = n // len(bin_values)
targets_per_bin = {}
deficit = 0  # unallocated pairs due to under-filled bins

# First pass – try to assign the base count to every bin
for b in bin_values:
    cap = capacity_per_bin[b]
    alloc = min(base, cap)
    targets_per_bin[b] = alloc
    deficit += base - alloc  # positive if we couldn't fill fully

# Second pass – greedily give remaining quotas to bins with spare capacity
spare_bins = sorted(bin_values, key=lambda x: capacity_per_bin[x] - targets_per_bin[x], reverse=True)
for b in spare_bins:
    if deficit == 0:
        break
    spare = capacity_per_bin[b] - targets_per_bin[b]
    if spare <= 0:
        continue
    take = min(spare, deficit)
    targets_per_bin[b] += take
    deficit -= take

# If still deficit, fallback proportional to remaining capacity (should be rare)
if deficit > 0:
    for b in spare_bins:
        if deficit == 0:
            break
        spare = capacity_per_bin[b] - targets_per_bin[b]
        if spare <= 0:
            continue
        take = min(spare, deficit)
        targets_per_bin[b] += take
        deficit -= take

assert sum(targets_per_bin.values()) == n, "Allocation bug: total assigned pairs ≠ n"

# Perform the bin-aware sampling
sample_normals = []
sample_abnormals = []
for b, t in targets_per_bin.items():
    if t == 0:
        continue
    norm_bin = normal_df[normal_df["age_bin"] == b].sample(n=t, random_state=42)
    abn_bin  = abnormal_df[abnormal_df["age_bin"] == b].sample(n=t, random_state=42)
    sample_normals.append(norm_bin)
    sample_abnormals.append(abn_bin)

normal_sample   = pd.concat(sample_normals, ignore_index=True)
abnormal_sample = pd.concat(sample_abnormals, ignore_index=True)

# Use combined normal and abnormal samples
filtered = pd.concat([normal_sample, abnormal_sample], ignore_index=True)

#show all labels
# print("Available labels in the dataset:")
# print(df.columns.tolist())

#print("Unique values in 'awake':", df["awake"].dropna().unique())
#print("Unique values in 'seizure':", df["seizure"].dropna().unique())
#print("Unique values in 'uninterpretable':", df["uninterpretable"].dropna().unique())

# Build S3 path for each session
def build_s3_path(row):
    return f"{row['SiteID']}/{row['BidsFolder']}/ses-{row['SessionID']}"

filtered.loc[:, "S3_Session_Path"] = filtered.apply(build_s3_path, axis=1)
print(f"Number of filtered EEG sessions: {len(filtered)}")

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_session(row):
    subject = row["BidsFolder"]
    session = f"ses-{row['SessionID']}"
    s3_path = f"s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-eeg-access-point/EEG/bids/{row['SiteID']}/{subject}/{session}/"
    local_path = os.path.join(desired_dir, f"bids_{n}_normal_abnormal", subject, session)

    try:
        subprocess.run(["mkdir", "-p", local_path], check=True)
        subprocess.run(["aws", "s3", "cp", s3_path, local_path, "--recursive"], check=True)
        return f"✅ Downloaded: {subject}/{session}"
    except subprocess.CalledProcessError as e:
        return f"❌ Failed: {subject}/{session} — {e}"

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(download_session, row) for _, row in filtered.iterrows()]
    for future in as_completed(futures):
        print(future.result())