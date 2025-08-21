

import os
import mne
import tqdm
from collections import Counter

# This script scans EDF files and counts how often each channel name appears.

def find_edf_files(root):
    """Recursively find all .edf files under root."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".edf"):
                yield os.path.join(dirpath, fn)

def check_channels(edf_path, observed_counts):
    """Increment observed counts for all channel names present in one EDF file."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        chs = raw.ch_names
        for ch in chs:
            observed_counts[ch] += 1
    except Exception as e:
        print(f"[ERROR] {edf_path}: {e}")

def summarize_observed(root):
    """Process all EDF files under root and return counts of observed channel names."""
    observed_counts = Counter()
    edf_files = list(find_edf_files(root))
    for edf_file in tqdm.tqdm(edf_files, total=len(edf_files), desc="Processing raws", unit="raw"):
        check_channels(edf_file, observed_counts)
    return observed_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check EDF channel name consistency")
    parser.add_argument("root", help="Root directory to search for EDF files")
    args = parser.parse_args()

    counts = summarize_observed(args.root)
    print("=== Channel occurrence summary (across all files) ===")
    for ch, cnt in counts.most_common():
        print(f"{ch}: {cnt}")