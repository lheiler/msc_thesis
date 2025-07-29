


import numpy as np
import multiprocessing as mp

# Ensure fork start method so local functions don't need pickling (macOS default is spawn)
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass  # start method already set

# ---------------------------------------------------------------------------
# Temporary workaround: pypet (dependency of neurolib) expects `np.string_`,
# which was removed in NumPy 2.0. Re-introduce alias before importing neurolib.
# ---------------------------------------------------------------------------
# Restore removed aliases expected by older libs (pypet/neurolib)
_alias_mappings = {
    "string_": "bytes_",
    "float_": "float64",
    "complex_": "complex128",
    "int_": "int64",
}
for alias, target in _alias_mappings.items():
    if not hasattr(np, alias):
        setattr(np, alias, getattr(np, target))

from neurolib.models.aln import ALNModel
from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace

import mne
import argparse
from scipy.signal import welch
import os

# Global reference for target PSD and frequency array
TARGET_PSD = None
FREQS = None


# ------------------------------------
# Fitness function (top-level picklable)
# ------------------------------------


def fitness_function(params):
    """Return negative MSE between simulated and empirical PSD."""
    global TARGET_PSD, FREQS

    model = ALNModel()
    model.params.update(params)
    model.run()

    if model.outputs is None or model[model.default_output] is None:
        return -np.inf

    sim_signal = model[model.default_output].squeeze()  # (N,) per node but we take first
    from scipy.signal import welch

    sim_freqs, sim_power = welch(sim_signal, fs=int(1000 / model.params["dt"]))
    sim_interp = np.interp(FREQS, sim_freqs, sim_power)
    return -float(np.mean((sim_interp - TARGET_PSD) ** 2))


# -----------------------------------------------------------------------------
# EEG dataset loader (same logic as prototype_ctm_coupled)
# -----------------------------------------------------------------------------

class EEGDataset:
    """Minimal re-implementation of the EEGDataset used in prototype_ctm_coupled.

    It supports two layouts:

    1. Pre-computed NumPy blobs::
           data_root/
             ├─ segments.npy   (N, 19, 7680)
             └─ labels.npy     (ignored here)

    2. Hierarchical folder with *.fif* files::
           data_root/
             ├─ normal/   *.fif
             └─ abnormal/ *.fif
    """

    SEG_LEN = 60 * 128  # 60 s @ 128 Hz

    def __init__(self, data_root: str):
        self.data_root = data_root

        seg_path = os.path.join(data_root, "segments.npy")
        if os.path.exists(seg_path):
            self.blob_mode = True
            self.segments = np.load(seg_path)  # (N, 19, 7680)
            return

        # else fall back to individual .fif segments
        self.blob_mode = False
        self.file_paths = []
        for root, _, files in os.walk(data_root):
            for fname in files:
                if fname.lower().endswith(".fif"):
                    self.file_paths.append(os.path.join(root, fname))
        if not self.file_paths:
            raise FileNotFoundError("No segments.npy or .fif files found in data_root")

    # ------------------------------------------------------------------
    def _load_fif_segment(self, path: str) -> np.ndarray:
        ch_names = [
            "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
        ]
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        raw.pick(ch_names)
        if raw.info["sfreq"] != 128.0:
            raw.resample(128.0)
        raw.crop(tmin=0, tmax=60-1/128)
        data = raw.get_data()  # (19, n_samples)
        n_samples = data.shape[1]
        if n_samples < self.SEG_LEN:
            padded = np.zeros((19, self.SEG_LEN), dtype=np.float32)
            padded[:, :n_samples] = data
            return padded
        return data[:, : self.SEG_LEN].astype(np.float32)

    # ------------------------------------------------------------------
    def iter_segments(self):
        """Yield (19,7680) arrays one by one."""
        if self.blob_mode:
            for seg in self.segments:
                yield seg
        else:
            for fp in self.file_paths:
                yield self._load_fif_segment(fp)

    def get_all_segments(self) -> np.ndarray:
        if not self.blob_mode:
            return np.stack(list(self.iter_segments()))
        return self.segments


# -----------------------------------------------------------------------------
# Helper: compute average PSD across dataset
# -----------------------------------------------------------------------------


def compute_dataset_psd(dataset: 'EEGDataset', fs: int = 128, n_fft: int = 512):
    """Compute averaged power spectrum across dataset
    dataset: EEGDataset instance
    """

    psd_accum = None
    n_total = 0
    for seg in dataset.iter_segments():
        for ch in range(seg.shape[0]):
            f, Pxx = welch(seg[ch], fs=fs, nperseg=n_fft)
            if psd_accum is None:
                psd_accum = np.zeros_like(Pxx)
            psd_accum += Pxx
        n_total += seg.shape[0]
    psd_avg = psd_accum / n_total
    return f, psd_avg  # type: ignore


# -----------------------------------------------------------------------------
# Main fitting routine
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fit neurolib ALN model to EEG spectra")
    parser.add_argument("--data_root", type=str, required=True, help="Path to EEG dataset root")
    args = parser.parse_args()

    # ---------------- Load data identically to prototype_ctm ----------------
    ds = EEGDataset(args.data_root)
    print(f"Loaded {len(ds.file_paths)} files")
    freqs, target_power = compute_dataset_psd(ds)

    # set globals for fitness function
    global TARGET_PSD, FREQS
    TARGET_PSD = target_power
    FREQS = freqs

    n_segments = ds.segments.shape[0] if getattr(ds, "blob_mode", False) else len(ds.file_paths)
    print(f"Loaded {n_segments} segments – target PSD computed over {len(freqs)} frequency bins.")

    # ---------------- Set up neurolib model & optimiser ----------------
    model = ALNModel()
    model.params["duration"] = 10000  # ms
    model.params["dt"] = 0.1

    parameter_bounds = {
        "mue_ext_mean": [0.5, 2.5],
        "mui_ext_mean": [0.2, 1.5],
        "Ke_gl": [200, 500],
        "Ki_gl": [50, 300],
    }

    parameter_space = ParameterSpace(parameter_bounds, kind="bound")

    evolution = Evolution(
        fitness_function,                # evalFunction
        parameter_space,                # ParameterSpace object
        model=model,
        ncores=4,                       # use serial evaluation to avoid pickling issues
        POP_INIT_SIZE=20,
        POP_SIZE=20,
        NGEN=10,
    )

    results = evolution.run()
    best_params = results.best_params
    np.save("best_eeg_params.npy", best_params)
    print("Best parameters found:", best_params)


if __name__ == "__main__":
    main()
# NOTE: Replace 'target_eeg_power.npy' with your real EEG-derived feature (e.g., channel-wise power spectra)