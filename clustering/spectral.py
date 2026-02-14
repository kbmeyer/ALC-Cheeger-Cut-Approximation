import numpy as np
from scipy.spatial.distance import cdist
import scipy.linalg as la

# ---------------- Distance matrix ----------------
def dist_mat(X):
    """Compute pairwise Euclidean distances (N x N)."""
    return cdist(X, X, "euclidean")

# ---------------- Row-wise (Ng–Jordan–Weiss) normalisation ----------------
def renorm_rows(x,eps=1e-12):
    norms = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    return x / np.maximum(norms, eps)

# ---------------- Additional metric for rank proxy experiment ----------------

def degree_heterogeneity_from_dist(X, sigma, zero_diagonal=True, eps=1e-12):
    D = dist_mat(X)
    d2 = D**2
    W = np.exp(-d2 / (2.0 * sigma**2 + eps))
    if zero_diagonal:
        np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    cv = deg.std(ddof=0) / (deg.mean() + eps)
    return deg, cv

# === NEW ===
def cluster_balance_stats(y_pred, eps=1e-12):
    """
    Balance diagnostics for a clustering:
      - min/max cluster fraction
      - entropy of cluster-size distribution
      - Gini coefficient of cluster sizes (optional but useful)
    """
    y = np.asarray(y_pred)
    _, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv)
    n = counts.sum()

    p = counts / (n + eps)
    min_frac = float(p.min()) if len(p) else np.nan
    max_frac = float(p.max()) if len(p) else np.nan

    entropy = float(-(p * np.log(p + eps)).sum()) if len(p) else np.nan

    # Gini over counts (scale-invariant measure of imbalance)
    c = np.sort(counts.astype(float))
    k = len(c)
    if k <= 1:
        gini = 0.0
    else:
        gini = float((2.0 * np.sum((np.arange(1, k + 1)) * c)) / (k * np.sum(c) + eps) - (k + 1) / k)

    return {
        "min_cluster_frac": min_frac,
        "max_cluster_frac": max_frac,
        "cluster_entropy": entropy,
        "cluster_gini": gini,
    }

# ---------------- Spectral decomposition (full Gaussian affinity) ----------------
def KM_spec_decom(sim_mat, sigma, Laplacian, I_mat):
    """
    Build Gaussian affinity and compute Laplacian eigendecomposition.

    Parameters
    ----------
    sim_mat : (N,N) ndarray
        Pairwise distances.
    sigma : float
        Gaussian kernel width.
    Laplacian : {"RW","SYM"}
        Random-walk or symmetric normalised Laplacian.
    I_mat : (N,N) ndarray
        Identity matrix.

    Returns
    -------
    eig_vals : (N) ndarray (ascending)
    eig_vecs : (N,N) ndarray (columns are eigenvectors)
    A        : (N,N) ndarray (Gaussian affinity)
    """
    A = np.exp(-sim_mat ** 2 / (2 * sigma ** 2))

    if Laplacian.upper() == "RW":
        D_1 = np.diag(1 / np.sum(A, axis=1))
        L = I_mat - D_1 @ A
        eig_vals, eig_vecs = la.eigh(L)

    elif Laplacian.upper() == "SYM":
        D_2 = np.diag(1 / np.sqrt(np.sum(A, axis=1)))  # D^{-1/2}
        L_sym = I_mat - D_2 @ A @ D_2
        eig_vals, eig_vecs = la.eigh(L_sym)  # SYM eigendecomposition
        

    else:
        raise ValueError("Laplacian must be 'RW' or 'SYM'")

    return eig_vals, eig_vecs, A

# ---------------- Main spectral embedding (backward compatible) ----------------
def spectral_embed(
    X,  # data from generators
    sigma,  # kernel width parameter
    k,  # number of clusters
    q,  # number of eigenvectors used
    Laplacian,
    seed=6,
):
    """
    Full-affinity spectral embedding (no kNN). Compatible with framework.
    Row-normalisation is applied only when using the symmetric Laplacian to follow Ng scheme.

    Parameters
    ----------
    X : (N, d) ndarray
        Input data.
    sigma : float
        Gaussian kernel width.
    k : int
        Target number of clusters (used if q is None).
    q : int or None
        Embedding dimension (default: k).
    laplacian_kind : {"sym", "rw"}
        Type of Laplacian.
    seed : int
        Random seed.


    Returns
    -------
    Y : (N, q) ndarray
        Spectral embedding (row-normalised only for L_sym).
    eig_vals : (N) ndarray
        Eigenvalues (ascending order).
    [A, S] : optional
        Gaussian affinity and distance matrices.
    """
    np.random.seed(seed)
    N = X.shape[0]
    I = np.eye(N)
  

    # Compute pairwise distances and perform spectral decomposition
    S = dist_mat(X)
    eig_vals, eig_vecs, A = KM_spec_decom(S, sigma, Laplacian, I)

    # Select embedding dimension
    q_eff = q if q is not None else k

    # Sort eigenpairs by eigenvalue
    idx = np.argsort(eig_vals)
    eig_vals = eig_vals[idx]
    Y = eig_vecs[:, :q_eff]

    if Laplacian == "SYM":
        Y = renorm_rows(Y)
        return Y, eig_vals
    else:
        return Y, eig_vals
