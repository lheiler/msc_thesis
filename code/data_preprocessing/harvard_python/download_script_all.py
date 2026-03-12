import pandas as pd

# Load EEG metadata
meta1 = pd.read_csv("/rds/general/user/lrh24/home/thesis/Datasets/harvard-eeg/eeg_metadata/S0001_eeg_metadata_2025-05-06.csv")
meta2 = pd.read_csv("/rds/general/user/lrh24/home/thesis/Datasets/harvard-eeg/eeg_metadata/S0002_eeg_metadata_2025-05-06.csv")

# Load EEG reports
reports1 = pd.read_csv("/rds/general/user/lrh24/home/thesis/Datasets/harvard-eeg/heedb_metadata/S0001_EEG__reports_findings.csv")
reports2 = pd.read_csv("/rds/general/user/lrh24/home/thesis/Datasets/harvard-eeg/heedb_metadata/S0002_EEG__reports_findings.csv")

# Combine metadata and reports
meta = pd.concat([meta1, meta2], ignore_index=True)
reports = pd.concat([reports1, reports2], ignore_index=True)

# Calculate duration in minutes
meta["DurationMinutes"] = meta["DurationInSeconds"] / 60

# Merge safely on 3 keys
df = pd.merge(meta, reports, on=["SiteID", "BDSPPatientID", "SessionID"], how="inner")

#show all labels
# print("Available labels in the dataset:")
# print(df.columns.tolist())

print("Unique values in 'awake':", df["awake"].dropna().unique())
print("Unique values in 'seizure':", df["seizure"].dropna().unique())
print("Unique values in 'uninterpretable':", df["uninterpretable"].dropna().unique())

# Filter for high-quality recordings
filtered = df[
    df["awake"].isin(["annotation", "report", "report annotation"]) &
    (~df["seizure"].isin([
        "report", "annotation", "verified",
        "report verified", "report annotation", "report annotation verified", "annotation verified"
    ])) &
    (df["uninterpretable"].isna()) &
    (df["DurationMinutes"].between(1, 100))
]

# Build S3 path for each session
def build_s3_path(row):
    return f"{row['SiteID']}/{row['BidsFolder']}/ses-{row['SessionID']}"

filtered.loc[:, "S3_Session_Path"] = filtered.apply(build_s3_path, axis=1)
print(f"Number of filtered EEG sessions: {len(filtered)}")
# Generate download shell script
with open("download_commands.txt", "w") as f:
    for _, row in filtered.iterrows():
        subject = row["BidsFolder"]
        session = f"ses-{row['SessionID']}"
        s3_path = f"s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-eeg-access-point/EEG/bids/{row['SiteID']}/{subject}/{session}/"
        local_path = f"./bids_root/{subject}/{session}/"
        edf_file = f"{local_path}eeg/{subject}_{session}_task-EEG_eeg.edf"

        command = f"[ ! -d {local_path} ] && mkdir -p {local_path} && aws s3 cp {s3_path} {local_path} --recursive \n"
        f.write(command)