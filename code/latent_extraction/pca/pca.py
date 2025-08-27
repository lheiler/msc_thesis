
from __future__ import annotations
import json
from typing import Optional, Union, List, Iterable
from pathlib import Path
import pickle
import numpy as np
import mne
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import sys
sys.path.append("/rds/general/user/lrh24/home/thesis/code")
from utils.util import compute_psd_from_raw, PSD_CALCULATION_PARAMS, STANDARD_EEG_CHANNELS


# Shared helpers / cleaning

def iter_fif_files(root: Path) -> Iterable[Path]:
    """Yield all .fif files recursively under root."""
    for p in root.rglob("*.fif"):
        if p.is_file():
            yield p


def _parse_n_components(s: str) -> Union[int, float, str]:
    """Deprecated: no longer used since we hardcode config in main()."""
    return int(s)


# def fit_pca_from_fif_dir(
#     train_dir: Union[str, Path],
#     model_out: Union[str, Path],
#     n_components: Union[int, float, str] = 0.95,
#     whiten: bool = False,
#     impute: Optional[str] = None,  # "median" | "mean" | None
#     preload: bool = True,
#     verbose: bool = True,
#     psd_n_fft: Optional[int] = None,
#     # ---- New PSD params (used for method=avg_psd) ----
#     psd_fmin: float = 1.0,
#     psd_fmax: float = 45.0,
#     psd_df: float = 1.0,
#     psd_n_per_seg: Optional[int] = 512,
# ) -> dict:
#     """
#     Recursively loads .fif files from train_dir, extracts per-channel PSDs, fits
#     StandardScaler + PCA on TRAIN ONLY, and saves a frozen artifact (.npz + .pkl).

#     """
#     train_dir = Path(train_dir)
#     model_out = Path(model_out)
#     files = list(iter_fif_files(train_dir))
#     if not files:
#         raise FileNotFoundError(f"No .fif files found under {train_dir}")

#     X_list: List[np.ndarray] = []
#     used_files: List[str] = []
#     first_dim: Optional[int] = None
#     freqs_ref: Optional[np.ndarray] = None

#     for fp in files:
#         try:
#             raw = mne.io.read_raw_fif(str(fp), preload=preload, verbose="ERROR")
#             psd = compute_psd_from_raw(raw, calculate_average=False, normalize=True)  # (C, F)
#             if not np.all(np.isfinite(psd)):
#                 if verbose:
#                     print(f"⚠️  Non-finite PSD; will handle via imputation/scaling: {fp}")
#             for ch_vec in psd:
#                 X_list.append(ch_vec.astype(np.float32))
#                 used_files.append(str(fp))
#         except Exception as e:
#             if verbose:
#                 print(f"⚠️  Failed on {fp}: {e}")
#             continue
    
#     if not X_list:
#         raise RuntimeError("No valid feature vectors were produced.")

#     X = np.vstack(X_list).astype(np.float32)  # (n, d)
#     d = X.shape[1]


#     # Scale + PCA
#     scaler = StandardScaler(with_mean=True, with_std=True)
#     Xs = scaler.fit_transform(X)

#     pca = PCA(n_components=n_components, whiten=whiten, svd_solver="auto", random_state=0)
#     Z = pca.fit_transform(Xs)

#     # Save light artifact for fast runtime + a .pkl with full objects (optional)
#     payload_npz = dict(
#         mu=scaler.mean_.astype(np.float32),
#         sigma=scaler.scale_.astype(np.float32),
#         V=pca.components_.astype(np.float32),           # (k, d)
#         lam=pca.explained_variance_.astype(np.float32), # (k,)
#         whiten=np.array(whiten, dtype=np.bool_),
#         input_dim=np.array([d], dtype=np.int32),
#         channels=np.array(STANDARD_EEG_CHANNELS, dtype=object),
#     )
#     model_out.parent.mkdir(parents=True, exist_ok=True)
#     np.savez(model_out, **payload_npz)

#     summary = {
#         "files_total": len(files),
#         "files_used": len(used_files),
#         "input_dim": d,
#         "k": int(pca.components_.shape[0]),
#         "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
#         "model_out_npz": str(model_out),
#         "model_out_pkl": str(model_out.with_suffix(".pkl")),
#     }
#     if verbose:
#         print(json.dumps(summary, indent=2))
#     return summary

# ---------------------------------------------------------------------
# New: Fit PCA from cleaned epoch pickle (train_epochs.pkl)
# ---------------------------------------------------------------------
def fit_pca_from_pickle(
    train_pickle: Union[str, Path],
    model_out: Union[str, Path],
    n_components: Union[int, float, str] = 0.95,
    whiten: bool = False,
    preload: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Load cleaned epochs from a single pickle file (list of 5‑tuples:
    (raw, gender, age, abnormal, sample_id)), compute per‑channel PSDs,
    fit StandardScaler + PCA on TRAIN ONLY, and save a frozen artifact (.npz).
    """
    train_pickle = Path(train_pickle)
    model_out = Path(model_out)

    if not train_pickle.exists():
        raise FileNotFoundError(f"Pickle file not found: {train_pickle}")

    with open(train_pickle, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError(f"No records found in {train_pickle}")

    X_list: List[np.ndarray] = []
    used_count: int = 0

    for rec in records:
        try:
            # Expecting (raw, g, a, ab, sample_id)
            raw = rec[0]
            psd = compute_psd_from_raw(raw, calculate_average=False, normalize=True)  # (C, F)
            if not np.all(np.isfinite(psd)):
                if verbose:
                    print("⚠️  Non-finite PSD encountered – sanitizing and continuing …")
                psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
            for ch_vec in psd:
                X_list.append(ch_vec.astype(np.float32))
            used_count += 1
        except Exception as e:
            if verbose:
                print(f"⚠️  Skipping record due to error: {e}")
            continue

    if not X_list:
        raise RuntimeError("No valid feature vectors were produced from pickle.")

    X = np.vstack(X_list).astype(np.float32)  # (n, d)
    d = X.shape[1]

    # Scale + PCA (train-only statistics)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="auto", random_state=0)
    _ = pca.fit_transform(Xs)

    payload_npz = dict(
        mu=scaler.mean_.astype(np.float32),
        sigma=scaler.scale_.astype(np.float32),
        V=pca.components_.astype(np.float32),           # (k, d)
        lam=pca.explained_variance_.astype(np.float32), # (k,)
        whiten=np.array(whiten, dtype=np.bool_),
        input_dim=np.array([d], dtype=np.int32),
        channels=np.array(STANDARD_EEG_CHANNELS, dtype=object),
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_out, **payload_npz)

    summary = {
        "records_used": int(used_count),
        "vectors": int(X.shape[0]),
        "input_dim": d,
        "k": int(pca.components_.shape[0]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "model_out_npz": str(model_out),
        "model_out_pkl": str(model_out.with_suffix(".pkl")),
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary
# -----------------------------
# Online: lightweight runtime
# -----------------------------
class FrozenPCA:
    """NumPy runtime transformer. Load once, transform many."""
    def __init__(self, npz_path: Union[str, Path]):
        blob = np.load(str(npz_path), allow_pickle=True)
        self.mu    = np.asarray(blob["mu"], dtype=np.float32)       # (d,)
        self.sigma = np.asarray(blob["sigma"], dtype=np.float32)    # (d,)
        self.V     = np.asarray(blob["V"], dtype=np.float32)        # (k, d)
        self.lam   = np.asarray(blob["lam"], dtype=np.float32)      # (k,)
        self.whiten = bool(blob["whiten"])
        self.input_dim = int(np.asarray(blob["input_dim"])[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        assert X.shape[1] == self.input_dim, f"Feature dim mismatch: {X.shape[1]} vs {self.input_dim}"
        Xs = (X - self.mu) / self.sigma
        Z  = Xs @ self.V.T
        if self.whiten:
            Z = Z / np.sqrt(self.lam)
        return Z

class FrozenPCATorch:
    """PyTorch runtime transformer. Load once, transform many (on CPU/GPU)."""
    def __init__(self, npz_path: Union[str, Path], device=None):
        import torch
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        blob = np.load(str(npz_path), allow_pickle=True)
        self.device = device
        self.mu    = torch.tensor(blob["mu"],    device=device, dtype=torch.float32)
        self.sigma = torch.tensor(blob["sigma"], device=device, dtype=torch.float32)
        self.V     = torch.tensor(blob["V"],     device=device, dtype=torch.float32)  # (k, d)
        self.lam   = torch.tensor(blob["lam"],   device=device, dtype=torch.float32)
        self.whiten = bool(blob["whiten"])
        self.input_dim = int(np.asarray(blob["input_dim"])[0])

    @property
    def k(self) -> int:
        return int(self.V.shape[0])

    @torch.no_grad()
    def transform_vec(self, x_vec) -> "torch.Tensor":
        """
        x_vec: 1D (d,) or 2D (n,d) tensor/array-like
        returns: (k,) or (n,k) torch.Tensor
        """
        import torch
        x = torch.as_tensor(x_vec, device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.input_dim, f"Feature dim mismatch: {x.shape[1]} vs {self.input_dim}"
        x = (x - self.mu) / self.sigma
        z = x @ self.V.t()
        if self.whiten:
            z = z / self.lam.sqrt()
        return z.squeeze(0) if z.shape[0] == 1 else z

def extract_pca_from_raw(raw: mne.io.BaseRaw, *, model: FrozenPCATorch, device: str = "cuda", per_channel: bool = False) -> torch.Tensor:
    """Extract PCA features from raw data."""
    if per_channel:
        psd = compute_psd_from_raw(raw, calculate_average=False, normalize=True)
        return model.transform_vec(psd).cpu().numpy().flatten()
    else:
        psd = compute_psd_from_raw(raw, calculate_average=True, normalize=True)
        return model.transform_vec(psd).cpu().numpy().flatten()


def main():
    # Hardcoded configuration – no CLI args
    train_pickle = "/rds/general/user/lrh24/home/thesis/Datasets/tuh-eeg-ab-clean/train_epochs.pkl"
    out_dir = Path("/rds/general/user/lrh24/home/thesis/code/latent_extraction/pca/models/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA settings
    k = 8
    whiten = False
    impute = None
    preload = True
    verbose = True

    # PSD settings
    psd_fmin = PSD_CALCULATION_PARAMS.get("min_freq", 3.0)
    psd_fmax = PSD_CALCULATION_PARAMS.get("max_freq", 45.0)
    psd_df = PSD_CALCULATION_PARAMS.get("df", 1.0)
    psd_n_per_seg = PSD_CALCULATION_PARAMS.get("n_per_seg", 512)
    psd_n_fft = PSD_CALCULATION_PARAMS.get("n_fft", 512)

    model_stem = f"pca_pc_psd_k{k}"
    model_out = out_dir / f"{model_stem}.npz"

    # Use cleaned epoch pickle instead of scanning FIF files
    fit_pca_from_pickle(
        train_pickle=train_pickle,
        model_out=model_out,
        n_components=k,
        whiten=whiten,
        preload=preload,
        verbose=verbose,
    )

if __name__ == "__main__":
    main()