
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np


def hierarchical_grouping(centroids: np.ndarray,
                          W: np.ndarray,
                          n_groups: int = 4,
                          *,
                          alpha: float = 0.5,
                          metric: str = "euclidean",
                          linkage_method: str = "ward",
                          ) -> np.ndarray:
    """
    Jointly cluster parcels by spatial *and* connectivity similarity.

    Parameters
    ----------
    centroids : (n_parcels, 3) MNI coordinates (mm).
    W         : (n_parcels, n_parcels) structural weight matrix.
    n_groups  : desired number of clusters.
    alpha     : weight on the spatial term
                (0 → only connectivity, 1 → only spatial).
    """
    # ------- spatial distance, scaled 0‥1 ---------------------------------
    D_sp = squareform(pdist(centroids, metric=metric))
    D_sp /= D_sp.max() + 1e-12

    # ------- connectivity “distance”, scaled 0‥1 --------------------------
    # similarity = Pearson correlation of connectivity fingerprints
    C     = np.corrcoef(W)
    D_conn = 1.0 - C                           # similarity → distance 0‥2
    D_conn -= D_conn.min()
    D_conn /= D_conn.max() + 1e-12

    # ------- hybrid distance ---------------------------------------------
    D_hybrid = alpha * D_sp + (1.0 - alpha) * D_conn

    # ---------- ensure symmetry & clean diagonals --------------------
    D_hybrid = np.nan_to_num(D_hybrid)              # NEW: kill NaNs
    D_hybrid = 0.5 * (D_hybrid + D_hybrid.T)        # NEW: enforce symmetry
    np.fill_diagonal(D_hybrid, 0.0)                # NEW: zeros on the diag

    Z = linkage(squareform(D_hybrid), method=linkage_method)
    labels = fcluster(Z, t=n_groups, criterion="maxclust") - 1   # 0‥n_groups-1
    return labels.astype(int)