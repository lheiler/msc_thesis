import os
import json
import numpy as np
from typing import Dict, Optional

from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

from utils.latent_loading import load_latent_parameters_array
from typing import Tuple, List


def _center_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def _hsic_linear(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = _center_rows(X)
    Yc = _center_rows(Y)
    K = Xc @ Xc.T  # (n,n)
    L = Yc @ Yc.T  # (n,n)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    return float(np.sum(KH * LH))


def linear_cka(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Linear CKA similarity between two embeddings (rows = samples).

    Returns a value in [0, 1] where higher indicates greater similarity.
    """
    hsic_xy = _hsic_linear(Z1, Z2)
    hsic_xx = _hsic_linear(Z1, Z1)
    hsic_yy = _hsic_linear(Z2, Z2)
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    val = hsic_xy / denom
    # Clamp small numerical issues
    return float(np.clip(val, -1.0, 1.0))


def rbf_cka(Z1: np.ndarray, Z2: np.ndarray, gamma1: Optional[float] = None, gamma2: Optional[float] = None) -> float:
    """RBF-CKA using median heuristic for bandwidth if not provided.
    Slower than linear CKA; prefer linear unless non-linear similarity is required.
    """
    def _rbf_kernel(Z: np.ndarray, gamma: Optional[float]) -> np.ndarray:
        D2 = pairwise_distances(Z, metric="sqeuclidean")
        if gamma is None:
            # median heuristic on non-zero distances
            nz = D2[D2 > 0]
            if nz.size == 0:
                gamma_eff = 1.0
            else:
                gamma_eff = 1.0 / (2.0 * np.median(nz))
        else:
            gamma_eff = gamma
        return np.exp(-gamma_eff * D2)

    K = _rbf_kernel(Z1, gamma1)
    L = _rbf_kernel(Z2, gamma2)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic_xy = float(np.sum(KH * LH))
    hsic_xx = float(np.sum(KH * KH))
    hsic_yy = float(np.sum(LH * LH))
    val = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)
    return float(np.clip(val, -1.0, 1.0))


def cca_maxcorr(Z1: np.ndarray, Z2: np.ndarray, n_components: Optional[int] = None) -> float:
    """Maximum canonical correlation between two embeddings.

    Rows must align to samples. Uses sklearn CCA.
    """
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)
    n, d1 = Z1.shape
    _, d2 = Z2.shape
    if n < 2 or d1 == 0 or d2 == 0:
        return 0.0
    k = min(d1, d2, n - 1)
    if n_components is not None:
        k = min(k, int(n_components))
    if k < 1:
        return 0.0
    cca = CCA(n_components=k, max_iter=500)
    Xc = _center_rows(Z1)
    Yc = _center_rows(Z2)
    Xc, Yc = cca.fit_transform(Xc, Yc)
    # Each component pair is maximally correlated; take the largest
    # Compute per-component Pearson corr
    corrs = []
    for i in range(Xc.shape[1]):
        x = Xc[:, i]
        y = Yc[:, i]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.max(np.abs(corrs))) if corrs else 0.0


def distance_geometry_corr(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Correlation of pairwise distances (upper triangle) between two spaces."""
    D1 = pairwise_distances(Z1)
    D2 = pairwise_distances(Z2)
    iu = np.triu_indices_from(D1, k=1)
    a, b = D1[iu], D2[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def knn_jaccard_overlap(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """Average Jaccard overlap of k-NN sets between two spaces.

    Returns mean over samples of |N1 ∩ N2| / |N1 ∪ N2|, excluding self.
    """
    n = Z1.shape[0]
    if n <= 1 or k < 1:
        return 0.0
    k = min(k, n - 1)
    D1 = pairwise_distances(Z1)
    D2 = pairwise_distances(Z2)
    np.fill_diagonal(D1, np.inf)
    np.fill_diagonal(D2, np.inf)
    idx1 = np.argsort(D1, axis=1)[:, :k]
    idx2 = np.argsort(D2, axis=1)[:, :k]
    scores = []
    for i in range(n):
        s1 = set(idx1[i].tolist())
        s2 = set(idx2[i].tolist())
        inter = len(s1 & s2)
        union = len(s1 | s2)
        scores.append(inter / union if union > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def procrustes_disparity(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """Procrustes disparity (lower = more similar) after optimal scaling/rotation/translation."""
    try:
        A, B = Z1, Z2
        if A.shape[1] != B.shape[1]:
            k = min(A.shape[1], B.shape[1])
            if k < 1:
                return 0.0
            # Project both to common dimension via PCA fitted independently
            p1 = PCA(n_components=k, random_state=42)
            p2 = PCA(n_components=k, random_state=42)
            A = p1.fit_transform(A)
            B = p2.fit_transform(B)
        _, _, disparity = procrustes(A, B)
        return float(disparity)
    except Exception:
        return 0.0


def compute_pairwise_summary(Z1: np.ndarray, Z2: np.ndarray, *, k: int = 10) -> Dict[str, float]:
    """Convenience: compute a suite of pairwise similarity metrics.

    Assumes rows are aligned samples for both inputs.
    """
    return {
        "cka_linear": linear_cka(Z1, Z2),
        "cka_rbf": rbf_cka(Z1, Z2),
        "cca_maxcorr": cca_maxcorr(Z1, Z2),
        "dist_geom_corr": distance_geometry_corr(Z1, Z2),
        "knn_jaccard_k%d" % k: knn_jaccard_overlap(Z1, Z2, k=k),
        "procrustes_disparity": procrustes_disparity(Z1, Z2),
    }


def _dataloader_to_array(loader) -> Tuple[np.ndarray, List[str] | None]:
    # Loader.dataset is a list of tuples: (latent, ...)
    arr = []
    for sample in loader.dataset:
        z = sample[0]
        if hasattr(z, "detach"):
            z = z.detach().cpu().numpy()
        arr.append(np.asarray(z, dtype=np.float32).reshape(-1))
    Z = np.stack(arr, axis=0) if arr else np.zeros((0, 0), dtype=np.float32)
    # Optional IDs were attached at load time
    sample_ids = getattr(loader, "sample_ids", None)
    return Z, sample_ids


def _align_by_ids(ZL: np.ndarray, idsL: List[str], ZR: np.ndarray, idsR: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # IDs are required on both sides
    if not idsL or not idsR:
        raise ValueError("Missing sample IDs for alignment.")
    # Build index by id
    idxL = {sid: i for i, sid in enumerate(idsL)}
    idxR = {sid: i for i, sid in enumerate(idsR)}
    common = [sid for sid in idsL if sid in idxR]
    if not common:
        raise ValueError("No overlapping sample IDs between inputs; cannot align.")
    if len(common) < min(len(idsL), len(idsR)):
        print(f"⚠️  Aligning on {len(common)} common sample IDs (left={len(idsL)}, right={len(idsR)}).")
    left_idx = np.array([idxL[sid] for sid in common], dtype=int)
    right_idx = np.array([idxR[sid] for sid in common], dtype=int)
    return ZL[left_idx], ZR[right_idx]


def _load_latents_any(path: str, batch_size: int = 1024):
    path = os.path.expanduser(path)
    if path.endswith(".json"):
        return load_latent_parameters_array(path, batch_size=batch_size)
    else:
        print("ERROR: Unknown file type")
        return None


if __name__ == "__main__":
    # Hardcode your inputs here
    LEFT_PATH = "Results/expA/temp_latent_features_eval.json"   # ← change me
    RIGHT_PATH = "Results/expB/temp_latent_features_eval.json"  # ← change me
    K = 10
    SUBSAMPLE = 0           # 0 = no subsampling; otherwise pick N samples
    BATCH_SIZE = 2048
    OUTPUT_JSON = ""        # e.g., "Results/exp_compare/expA_vs_expB_eval.json"

    # Load
    left_loader = _load_latents_any(LEFT_PATH, batch_size=BATCH_SIZE)
    right_loader = _load_latents_any(RIGHT_PATH, batch_size=BATCH_SIZE)
    ZL, idsL = _dataloader_to_array(left_loader)
    ZR, idsR = _dataloader_to_array(right_loader)

    if ZL.shape[0] == 0 or ZR.shape[0] == 0:
        raise SystemExit("No samples loaded from one or both inputs.")

    # Align by sample IDs (required)
    if idsL is None or idsR is None:
        raise SystemExit("Latent loaders missing required sample IDs. Ensure JSONL includes sample_id.")
    ZL, ZR = _align_by_ids(ZL, idsL, ZR, idsR)
    n = ZL.shape[0]

    # Subsample if requested
    if SUBSAMPLE and n > SUBSAMPLE:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=SUBSAMPLE, replace=False)
        ZL = ZL[idx]
        ZR = ZR[idx]
        n = SUBSAMPLE

    scores = compute_pairwise_summary(ZL, ZR, k=K)
    print("Pairwise similarity scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    if OUTPUT_JSON:
        out_path = os.path.expanduser(OUTPUT_JSON)
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"Saved scores to {out_path}")


