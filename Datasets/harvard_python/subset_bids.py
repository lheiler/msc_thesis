import shutil
import random
from pathlib import Path

# === Configuration ===
root_dir = Path("/rds/general/user/lrh24/ephemeral/harvard-eeg/EEG")
full_bids = root_dir / "bids_root"
small_bids = root_dir / "bids_root_small"

# === Make output directory ===
small_bids.mkdir(exist_ok=True)

# === Get subject folders (sub-XXXX) ===
subject_dirs = [d for d in full_bids.glob("sub-*") if d.is_dir()]
subset_size = max(1, int(len(subject_dirs) * 0.1))

# === Random sample of subjects ===
subset_subjects = random.sample(subject_dirs, subset_size)
print(f"Copying {len(subset_subjects)} of {len(subject_dirs)} subjects to {small_bids}")

# === Copy each subject folder recursively ===
for subj_path in subset_subjects:
    dest = small_bids / subj_path.name
    shutil.copytree(subj_path, dest, dirs_exist_ok=True)

print("✅ Done! Subset created at:", small_bids)