"""Scan EDF files and count channel-name occurrences across a directory."""
from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Iterator

import mne
import tqdm

logger = logging.getLogger(__name__)


def find_edf_files(root: str) -> Iterator[str]:
    """Recursively find all ``.edf`` files under *root*."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".edf"):
                yield os.path.join(dirpath, fn)


def check_channels(edf_path: str, observed_counts: Counter) -> None:
    """Increment *observed_counts* for each channel name in one EDF file."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        for ch in raw.ch_names:
            observed_counts[ch] += 1
    except (OSError, ValueError) as exc:
        logger.error("Failed to read %s: %s", edf_path, exc)


def summarize_observed(root: str) -> Counter:
    """Process all EDF files under *root* and return channel-name counts."""
    observed_counts: Counter = Counter()
    edf_files = list(find_edf_files(root))
    for edf_file in tqdm.tqdm(
        edf_files, total=len(edf_files), desc="Processing raws", unit="raw",
    ):
        check_channels(edf_file, observed_counts)
    return observed_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check EDF channel name consistency",
    )
    parser.add_argument("root", help="Root directory to search for EDF files")
    args = parser.parse_args()

    counts = summarize_observed(args.root)
    print("=== Channel occurrence summary (across all files) ===")
    for ch, cnt in counts.most_common():
        print(f"{ch}: {cnt}")
