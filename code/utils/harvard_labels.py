from __future__ import annotations

# =============================================================================
# utils.harvard_labels
# -----------------------------------------------------------------------------
# Utility script to list **all** label values occurring in the Harvard-EEG
# metadata CSVs. This inspects the same CSV files that the data loader relies on
# and aggregates *unique* values per column.
#
# Usage (from repository root):
#     python -m utils.harvard_labels
#
# The output will be a nicely formatted overview of every column together with
# the distinct values it contains.
# =============================================================================

import textwrap
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd

# -----------------------------------------------------------------------------
# Hard-coded paths (adjust if your install moves!)
# -----------------------------------------------------------------------------

# Root folder containing the Harvard-EEG metadata CSVs
METADATA_DIR = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/metadata")

# Where the summary will be written (overwritten on each run)
OUTPUT_TXT = Path("/rds/general/user/lrh24/home/thesis/code/utils/harvard_labels_summary.txt")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _read_csvs(csv_paths: List[Path]) -> List[pd.DataFrame]:
    """Read CSVs while silencing irrelevant warnings."""
    dfs: List[pd.DataFrame] = []
    for fp in csv_paths:
        try:
            df = pd.read_csv(fp, low_memory=False)
            dfs.append(df)
        except Exception as exc:
            print(f"⚠️  Skipping {fp.name}: {exc}")
    return dfs


def _collect_unique_values(dfs: List[pd.DataFrame], columns: List[str]) -> Dict[str, Set[str]]:
    """Return a mapping *column → set(unique stringified values)*."""
    col_vals: Dict[str, Set[str]] = {c: set() for c in columns}

    for df in dfs:
        for col in columns:
            if col not in df.columns:
                continue
            # Convert each entry to a cleaned string representation
            unique_raw = df[col].unique()
            cleaned = {
                ("" if pd.isna(v) else str(v).strip())
                for v in unique_raw
            }
            col_vals[col].update(cleaned)

    return col_vals

# -----------------------------------------------------------------------------
# Label-column selection logic
#   We keep:
#     • explicit demographic / label fields we know are relevant (AGE, SEX, seizure)
#     • all abnormality-descriptor columns from ABNORMAL_COLS
#     • any *entirely lowercase* column name (these are the free-text label
#       categories such as "normal", "awake", "n1", "spikes", …)
# -----------------------------------------------------------------------------

MANDATORY_COLS = {"AgeAtVisit", "SexDSC", "seizure"}

ABNORMAL_COLS = [
    "spikes", "lpd", "gpd", "lrda", "grda", "bs",
    "foc slowing", "gen slowing", "bipd", "status"
]

def _is_label_column(col: str) -> bool:
    if col in MANDATORY_COLS or col in ABNORMAL_COLS:
        return True
    # All lowercase (excluding spaces) → likely one-word/phrase label
    return col.replace(" ", "") == col.replace(" ", "").lower()


# -----------------------------------------------------------------------------
# Main routine (no CLI interaction – paths are hard-coded)
# -----------------------------------------------------------------------------

def main() -> None:
    metadata_dir = METADATA_DIR
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"Metadata directory not found: {metadata_dir}\n"
            "Please adjust METADATA_DIR at the top of utils/harvard_labels.py."
        )

    # Find all relevant CSVs (mimics the loader's query patterns)
    csv_files  = list(metadata_dir.glob("*eeg_metadata_*.csv"))
    csv_files += list(metadata_dir.glob("*_EEG__reports_findings.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No matching metadata CSVs found in {metadata_dir}. "
            "Please double-check the path."
        )

    print(f"🔍  Found {len(csv_files)} metadata file(s). Parsing …")

    dfs = _read_csvs(csv_files)

    # ------------------------------------------------------------------
    # Collect *only* the relevant label columns according to the filter
    # ------------------------------------------------------------------
    label_cols_set: Set[str] = set()
    for df in dfs:
        for col in df.columns.astype(str):
            if _is_label_column(col):
                label_cols_set.add(col)

    common_cols: List[str] = sorted(label_cols_set)

    values_map = _collect_unique_values(dfs, common_cols)

    # Pretty print -------------------------------------------------------------
    wrap = lambda s: textwrap.fill(s, width=70, subsequent_indent=" " * 12)

    lines = ["=== Unique label values per column ===\n"]
    for col in common_cols:
        if not values_map[col]:
            continue  # column was not present in any CSV
        joined = ", ".join(sorted(values_map[col]))
        line = f"{col:>11}: {wrap(joined)}"
        print(line)
        lines.append(line)

    # ------------------------------------------------------------------
    # Persist results to text file
    # ------------------------------------------------------------------
    OUTPUT_TXT.write_text("\n".join(lines))
    print(f"\n💾  Summary saved to {OUTPUT_TXT}\n")


if __name__ == "__main__":
    main() 