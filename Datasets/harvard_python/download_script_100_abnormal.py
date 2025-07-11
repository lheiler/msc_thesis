import pandas as pd
import os

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

# Define abnormal EEGs excluding seizure-based ones
abnormal_df = df[
    (df[abnormal_cols].fillna("none").apply(lambda row: any(val not in ["none", "normal", "", None] for val in row), axis=1)) &
    (~df["seizure"].fillna("").str.contains("seizure", case=False))
]

if len(normal_df) < 100:
    raise ValueError(f"Not enough normal EEGs. Found {len(normal_df)}, need at least 100.")
if len(abnormal_df) < 100:
    raise ValueError(f"Not enough abnormal EEGs. Found {len(abnormal_df)}, need at least 100.")

# Sample matched sets
normal_sample = normal_df.sample(n=100, random_state=42)
abnormal_sample = abnormal_df.sample(n=100, random_state=42)

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
    local_path = os.path.join(desired_dir, "bids_100_normal_abnormal", subject, session)

    try:
        subprocess.run(["mkdir", "-p", local_path], check=True)
        subprocess.run(["aws", "s3", "cp", s3_path, local_path, "--recursive"], check=True)
        return f"✅ Downloaded: {subject}/{session}"
    except subprocess.CalledProcessError as e:
        return f"❌ Failed: {subject}/{session} — {e}"

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(download_session, row) for _, row in filtered.iterrows()]
    for future in as_completed(futures):
        print(future.result())