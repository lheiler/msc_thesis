
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Optional, Union, List, Iterable

import joblib
import numpy as np
import mne
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch


# -----------------------------
# Shared helpers / cleaning
# -----------------------------
EEG_19 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
          'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
          'Cz', 'Pz', 'Fz']

def clean_x(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Drop A1/A2 if present, then pick the standard 19 EEG channels."""
    if 'A1' in raw.ch_names:
        raw.drop_channels(['A1'])
    if 'A2' in raw.ch_names:
        raw.drop_channels(['A2'])
    raw.pick_channels(EEG_19)
    return raw

def iter_fif_files(root: Path) -> Iterable[Path]:
    """Yield all .fif files recursively under root."""
    for p in root.rglob("*.fif"):
        if p.is_file():
            yield p


def _parse_n_components(s: str) -> Union[int, float, str]:
    """Deprecated: no longer used since we hardcode config in main()."""
    return int(s)


def fit_pca_from_fif_dir(
    train_dir: Union[str, Path],
    model_out: Union[str, Path],
    n_components: Union[int, float, str] = 0.95,
    whiten: bool = False,
    impute: Optional[str] = None,  # "median" | "mean" | None
    preload: bool = True,
    verbose: bool = True,
    psd_n_fft: Optional[int] = None,
    # ---- New PSD params (used for method=avg_psd) ----
    psd_fmin: float = 1.0,
    psd_fmax: float = 45.0,
    psd_df: float = 1.0,
    psd_n_per_seg: Optional[int] = 512,
    psd_log: bool = True,
    psd_resample: Optional[float] = None,
) -> dict:
    """
    Recursively loads .fif files from train_dir, extracts the average PSD feature, fits
    StandardScaler + PCA on TRAIN ONLY, and saves a frozen artifact (.npz + .pkl).

    """
    train_dir = Path(train_dir)
    model_out = Path(model_out)
    files = list(iter_fif_files(train_dir))
    if not files:
        raise FileNotFoundError(f"No .fif files found under {train_dir}")

    X_list: List[np.ndarray] = []
    used_files: List[str] = []
    first_dim: Optional[int] = None
    freqs_ref: Optional[np.ndarray] = None

    for fp in files:
        try:
            raw = mne.io.read_raw_fif(str(fp), preload=preload, verbose="ERROR")
            raw = clean_x(raw)
            r = raw
            if psd_resample is not None:
                r = raw.copy().resample(psd_resample, npad="auto")
            psd = r.compute_psd(
                method="welch",
                fmin=psd_fmin,
                fmax=psd_fmax,
                n_per_seg=psd_n_per_seg,
                n_fft=psd_n_fft,
                verbose="ERROR",
            )
            psds, freqs = psd.get_data(return_freqs=True)
            avg_psd = psds.mean(axis=0)  # (n_freqs,)
            if psd_log:
                avg_psd = np.log10(np.maximum(avg_psd, 1e-12))

            if freqs_ref is None:
                freqs_ref = freqs
                first_dim = avg_psd.shape[0]
            else:
                if avg_psd.shape[0] != first_dim or not np.allclose(freqs, freqs_ref, atol=1e-6, rtol=0):
                    raise ValueError(
                        f"Frequency grid mismatch for {fp}.\n"
                        f"Expected n={first_dim} and freqs≈{freqs_ref[:5]}...{freqs_ref[-5:]},\n"
                        f"got n={avg_psd.shape[0]} and freqs≈{freqs[:5]}...{freqs[-5:]}.\n"
                        f"Fix by using --psd-resample and/or --psd-n-fft to enforce identical bins, "
                        f"or disable --no-interp."
                    )

            if not np.all(np.isfinite(avg_psd)):
                if verbose:
                    print(f"⚠️  Non-finite PSD; will handle via imputation/scaling: {fp}")

            X_list.append(avg_psd.astype(np.float32))
            used_files.append(str(fp))
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed on {fp}: {e}")
            continue
    
    if not X_list:
        raise RuntimeError("No valid feature vectors were produced.")

    X = np.vstack(X_list).astype(np.float32)  # (n, d)
    d = X.shape[1]


    # Scale + PCA
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="auto", random_state=0)
    Z = pca.fit_transform(Xs)

    # Save light artifact for fast runtime + a .pkl with full objects (optional)
    payload_npz = dict(
        mu=scaler.mean_.astype(np.float32),
        sigma=scaler.scale_.astype(np.float32),
        V=pca.components_.astype(np.float32),           # (k, d)
        lam=pca.explained_variance_.astype(np.float32), # (k,)
        whiten=np.array(whiten, dtype=np.bool_),
        input_dim=np.array([d], dtype=np.int32),
        channels=np.array(EEG_19, dtype=object),
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_out, **payload_npz)

    summary = {
        "files_total": len(files),
        "files_used": len(used_files),
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

def main():
    # Hardcoded configuration – no CLI args
    train_dir = "/homes/lrh24/thesis/Datasets/tuh-eeg-ab-clean/train"
    out_dir = Path("/homes/lrh24/thesis/code/latent_extraction/pca/models/")
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA settings
    k = 8
    whiten = False
    impute = None
    preload = True
    verbose = True

    # PSD settings
    psd_fmin = 1.0
    psd_fmax = 45.0
    psd_df = 1.0
    psd_n_per_seg = 512
    psd_n_fft = 512
    psd_log = True
    psd_resample = None

    model_stem = f"pca_avg_psd_k{k}"
    model_out = out_dir / f"{model_stem}.npz"

    fit_pca_from_fif_dir(
        train_dir=train_dir,
        model_out=model_out,
        n_components=k,
        whiten=whiten,
        impute=impute,
        preload=preload,
        verbose=verbose,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        psd_df=psd_df,
        psd_n_per_seg=psd_n_per_seg,
        psd_log=psd_log,
        psd_resample=psd_resample,
        psd_n_fft=psd_n_fft,
    )

if __name__ == "__main__":
    main()