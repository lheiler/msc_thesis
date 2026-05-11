"""PCA-based latent feature extraction for EEG power spectra.

Provides:
    * ``FrozenPCA`` / ``FrozenPCATorch`` - runtime transformers loaded from
      a frozen ``.npz`` artifact (mean, scale, components).
    * ``extract_pca_from_raw`` - one-liner to project a Raw's PSD into PCA space.
    * ``fit_pca_from_pickle`` - train-time fitting from a cleaned-epoch pickle.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import mne
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.util import (
    STANDARD_EEG_CHANNELS,
    PSD_CALCULATION_PARAMS,
    compute_psd_from_raw,
)

logger = logging.getLogger(__name__)


def iter_fif_files(root: Path):
    """Yield all ``.fif`` files recursively under *root*."""
    for p in root.rglob("*.fif"):
        if p.is_file():
            yield p


# ---------------------------------------------------------------------------
# Train-time: fit PCA from cleaned epoch pickle
# ---------------------------------------------------------------------------

def fit_pca_from_pickle(
    train_pickle: Union[str, Path],
    model_out: Union[str, Path],
    n_components: Union[int, float, str] = 0.95,
    whiten: bool = False,
    preload: bool = True,
    verbose: bool = True,
) -> dict:
    """Fit StandardScaler + PCA on per-channel PSDs from a pickle file.

    Args:
        train_pickle: Path to ``train_epochs.pkl`` (list of 5-tuples).
        model_out: Output ``.npz`` path for the frozen artifact.
        n_components: PCA components (int, variance fraction, or ``"mle"``).
        whiten: Whether to whiten the components.
        preload: Unused (kept for API compatibility).
        verbose: Whether to print progress.

    Returns:
        Summary dict with fitting statistics.
    """
    train_pickle = Path(train_pickle)
    model_out = Path(model_out)

    if not train_pickle.exists():
        raise FileNotFoundError(f"Pickle file not found: {train_pickle}")

    with open(train_pickle, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise RuntimeError(f"No records found in {train_pickle}")

    X_list: list[np.ndarray] = []
    used_count = 0

    for rec in records:
        try:
            raw = rec[0]
            psd = compute_psd_from_raw(raw, calculate_average=False, normalize=True)
            if not np.all(np.isfinite(psd)):
                logger.warning("Non-finite PSD encountered - sanitizing")
                psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
            for ch_vec in psd:
                X_list.append(ch_vec.astype(np.float32))
            used_count += 1
        except Exception as exc:
            if verbose:
                logger.warning("Skipping record due to error: %s", exc)
            continue

    if not X_list:
        raise RuntimeError("No valid feature vectors were produced from pickle.")

    X = np.vstack(X_list).astype(np.float32)
    d = X.shape[1]

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="auto", random_state=0)
    pca.fit_transform(Xs)

    payload_npz = dict(
        mu=scaler.mean_.astype(np.float32),
        sigma=scaler.scale_.astype(np.float32),
        V=pca.components_.astype(np.float32),
        lam=pca.explained_variance_.astype(np.float32),
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
    }
    if verbose:
        logger.info("PCA fitting summary:\n%s", json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Runtime: frozen PCA transformers
# ---------------------------------------------------------------------------

class FrozenPCA:
    """NumPy runtime PCA transformer loaded from a ``.npz`` artifact.

    Args:
        npz_path: Path to the saved ``.npz`` file.
    """

    def __init__(self, npz_path: Union[str, Path]) -> None:
        blob = np.load(str(npz_path), allow_pickle=True)
        self.mu = np.asarray(blob["mu"], dtype=np.float32)
        self.sigma = np.asarray(blob["sigma"], dtype=np.float32)
        self.V = np.asarray(blob["V"], dtype=np.float32)
        self.lam = np.asarray(blob["lam"], dtype=np.float32)
        self.whiten = bool(blob["whiten"])
        self.input_dim = int(np.asarray(blob["input_dim"])[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* into PCA space.

        Args:
            X: Input array ``(n, d)`` or ``(d,)``.

        Returns:
            Projected array ``(n, k)`` or ``(k,)``.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        assert X.shape[1] == self.input_dim, (
            f"Feature dim mismatch: {X.shape[1]} vs {self.input_dim}"
        )
        Xs = (X - self.mu) / self.sigma
        Z = Xs @ self.V.T
        if self.whiten:
            Z = Z / np.sqrt(self.lam)
        return Z


class FrozenPCATorch:
    """PyTorch runtime PCA transformer loaded from a ``.npz`` artifact.

    Args:
        npz_path: Path to the saved ``.npz`` file.
        device: Torch device (auto-detected if ``None``).
    """

    def __init__(
        self, npz_path: Union[str, Path], device: Optional[torch.device] = None
    ) -> None:
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
        blob = np.load(str(npz_path), allow_pickle=True)
        self.device = device
        self.mu = torch.tensor(blob["mu"], device=device, dtype=torch.float32)
        self.sigma = torch.tensor(blob["sigma"], device=device, dtype=torch.float32)
        self.V = torch.tensor(blob["V"], device=device, dtype=torch.float32)
        self.lam = torch.tensor(blob["lam"], device=device, dtype=torch.float32)
        self.whiten = bool(blob["whiten"])
        self.input_dim = int(np.asarray(blob["input_dim"])[0])

    @property
    def k(self) -> int:
        """Number of principal components."""
        return int(self.V.shape[0])

    @torch.no_grad()
    def transform_vec(self, x_vec: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Project *x_vec* into PCA space.

        Args:
            x_vec: 1-D ``(d,)`` or 2-D ``(n, d)`` input.

        Returns:
            ``(k,)`` or ``(n, k)`` projected tensor.
        """
        x = torch.as_tensor(x_vec, device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.shape[1] == self.input_dim, (
            f"Feature dim mismatch: {x.shape[1]} vs {self.input_dim}"
        )
        x = (x - self.mu) / self.sigma
        z = x @ self.V.t()
        if self.whiten:
            z = z / self.lam.sqrt()
        return z.squeeze(0) if z.shape[0] == 1 else z


def extract_pca_from_raw(
    raw: mne.io.BaseRaw,
    *,
    model: FrozenPCATorch,
    device: str = "cuda",
    per_channel: bool = False,
) -> np.ndarray:
    """Project PSD features from *raw* into PCA space.

    Args:
        raw: MNE Raw object.
        model: Frozen PCA transformer.
        device: Torch device string.
        per_channel: If True, project per channel; else on the average PSD.

    Returns:
        Flattened float32 array of PCA features.
    """
    psd = compute_psd_from_raw(
        raw, calculate_average=not per_channel, normalize=True,
    )
    return model.transform_vec(psd).cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# CLI entry point for PCA model fitting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit PCA on dataset")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to train_epochs.pkl",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="tuh",
        help="Dataset name prefix for the saved model",
    )
    args = parser.parse_args()

    out_dir = Path("latent_extraction/pca/models/")
    out_dir.mkdir(parents=True, exist_ok=True)

    N_COMPONENTS = 8
    model_out = out_dir / f"{args.dataset_name}_pca_pc_psd_k{N_COMPONENTS}.npz"

    fit_pca_from_pickle(
        train_pickle=args.data_root,
        model_out=model_out,
        n_components=N_COMPONENTS,
        whiten=False,
        verbose=True,
    )
