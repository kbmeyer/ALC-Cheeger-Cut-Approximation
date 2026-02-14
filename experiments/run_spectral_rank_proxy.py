# experiments/run_spectral_rank_proxy.py
# %%
"""
Spectral Rank as a Proxy (Eigenvector rank over-specification) runner.

This script generalises your notebook experiment to run end-to-end in Spyder
(or from terminal) for: blobs, circles, moons.

For each seed × sigma × laplacian × rank r:
  - compute spectral embedding E_r using q=r leading eigenvectors
  - cluster E_r with ALC (using your existing ALC adapter "as is")
  - compute ARI and cluster-balance diagnostics
Optionally logs eigenvalues + spectral gaps into df_checks.

Outputs:
  results/spectral_rank_proxy/<tag>_raw.csv
  results/spectral_rank_proxy/<tag>_agg.csv
  results/spectral_rank_proxy/<tag>_checks.csv (if log_eigs or compare_sym_rw)
  figures/spectral_rank_proxy/<tag>_ari_vs_rank.pdf
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.metrics import adjusted_rand_score

from utils.io import ensure_dir
from utils.seed import set_global_seed

from experiments.common import resolve_dataset  # blobs/circles/moons -> generator

# Use your existing spectral implementation
from clustering.spectral import spectral_embed, dist_mat

# Use your existing ALC adapter "as is"
from clustering.alc import ALCAdapter


# %%
# -----------------------------
# Diagnostics helpers (kept local to avoid dependency churn)
# -----------------------------
def degree_heterogeneity_from_dist(X: np.ndarray, sigma: float, zero_diagonal: bool = True, eps: float = 1e-12):
    """
    Degree stats of the Gaussian affinity graph at bandwidth sigma.
    Returns: (deg_vector, deg_stats_dict)
    """
    D = dist_mat(X)
    d2 = D**2
    W = np.exp(-d2 / (2.0 * sigma**2 + eps))
    if zero_diagonal:
        np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)

    mean = float(deg.mean())
    std = float(deg.std(ddof=0))
    cv = std / (mean + eps)

    return deg, {
        "deg_mean": mean,
        "deg_std": std,
        "deg_cv": float(cv),
        "deg_min": float(deg.min()),
        "deg_max": float(deg.max()),
        "deg_max_min_ratio": float(deg.max() / (deg.min() + eps)),
    }


def cluster_balance_stats(y_pred, eps: float = 1e-12):
    """
    Cluster balance diagnostics:
      - min/max cluster fraction
      - entropy of cluster-size distribution
      - Gini coefficient of cluster sizes
    """
    y = np.asarray(y_pred)
    _, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv)
    n = counts.sum()

    p = counts / (n + eps)
    min_frac = float(p.min()) if len(p) else np.nan
    max_frac = float(p.max()) if len(p) else np.nan
    entropy = float(-(p * np.log(p + eps)).sum()) if len(p) else np.nan

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

# %%
@dataclass
class Cfg:
    # Which dataset to run
    dataset: str = "blobs"                 # blobs | circles | moons

    # Graph + embedding settings
    sigma_graph: List[float] = None        # list of sigmas (sigma grid)
    laplacians: List[str] = None           # ["SYM","RW"] or whichever your spectral code expects
    ranks: List[int] = None                # ranks r (q=r eigenvectors)
    seeds: List[int] = None                # e.g. list(range(10))

    # ALC settings
    alc_method: str = "GM"
    cn: Optional[int] = None               # if None, defaults to n_clusters (truth)
    G_mode: str = "corr"

    # Diagnostics toggles
    log_eigs: bool = True
    compare_sym_rw: bool = False           # optionally compute SYM vs RW closeness checks

    # Output
    results_dir: str = "results/spectral_rank_proxy"
    figures_dir: str = "figures/spectral_rank_proxy"

    def __post_init__(self):
        if self.sigma_graph is None:
            self.sigma_graph = [0.35]
        if self.laplacians is None:
            self.laplacians = ["SYM", "RW"]
        if self.ranks is None:
            self.ranks = list(range(2, 26))  # typical over-spec sweep
        if self.seeds is None:
            self.seeds = list(range(10))


# %%
def run(cfg: Cfg) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], str]:
    """
    Runs the experiment and saves CSVs + a figure.
    Returns: (df_raw, df_agg, df_checks_or_None, fig_path)
    """
    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.figures_dir)

    maker, data_kwargs, n_clusters = resolve_dataset(cfg.dataset)
    cn = cfg.cn if cfg.cn is not None else n_clusters

    sigma_grid = [float(s) for s in cfg.sigma_graph]
    ranks = [int(r) for r in cfg.ranks]
    seeds = [int(s) for s in cfg.seeds]
    laplacians = [str(l).upper() for l in cfg.laplacians]

    max_r = int(max(ranks))
    eig_q = max_r + 1  # need r+1 to compute spectral gaps

    rows: List[Dict[str, Any]] = []
    checks_rows: List[Dict[str, Any]] = []

    tag = (
        f"{cfg.dataset}_sigmas{len(sigma_grid)}_laps{len(laplacians)}"
        f"_r{min(ranks)}to{max(ranks)}_seeds{len(seeds)}_{cfg.alc_method}"
    )
    raw_csv = os.path.join(cfg.results_dir, f"{tag}_raw.csv")
    agg_csv = os.path.join(cfg.results_dir, f"{tag}_agg.csv")
    checks_csv = os.path.join(cfg.results_dir, f"{tag}_checks.csv")

    total = len(seeds) * len(sigma_grid) * len(laplacians) * len(ranks)
    
    alc = ALCAdapter()

    with tqdm(total=total, desc="Spectral Rank Proxy", leave=True) as pbar:
        for seed in seeds:
            set_global_seed(seed)

            X, y_true = maker(seed=seed, **data_kwargs)
            X = np.asarray(X)
            y_true = np.asarray(y_true)

            for sig in sigma_grid:
                # Graph diagnostics per (seed, sigma)
                _, deg_stats = degree_heterogeneity_from_dist(X, sigma=sig)

                for lap in laplacians:
                    # Cache eigenvalues once per (seed, sigma, lap) if needed
                    eigvals_cached = None
                    if cfg.log_eigs or cfg.compare_sym_rw:
                        _E_tmp, eigvals_cached = spectral_embed(
                            X=X, sigma=sig, k=n_clusters, q=eig_q, Laplacian=lap, seed=seed
                        )
                        eigvals_cached = np.asarray(eigvals_cached).ravel()

                    for r in ranks:
                        # Optional SYM vs RW comparison checks
                        if cfg.compare_sym_rw:
                            E_sym, vals_sym = spectral_embed(X=X, sigma=sig, k=n_clusters, q=r, Laplacian="SYM", seed=seed)
                            E_rw,  vals_rw  = spectral_embed(X=X, sigma=sig, k=n_clusters, q=r, Laplacian="RW",  seed=seed)

                            eigvals_close = np.allclose(vals_sym[:r], vals_rw[:r], rtol=1e-6, atol=1e-8)
                            E_close = np.allclose(E_sym, E_rw, rtol=1e-6, atol=1e-8)

                            # Principal angles via QR + SVD
                            Q_sym, _ = np.linalg.qr(E_sym)
                            Q_rw, _ = np.linalg.qr(E_rw)
                            S = np.linalg.svd(Q_sym.T @ Q_rw, compute_uv=False)
                            min_cos_angle = float(S.min())

                            from sklearn.metrics import pairwise_distances
                            D_sym = pairwise_distances(E_sym)
                            D_rw  = pairwise_distances(E_rw)
                            distances_close = np.allclose(D_sym, D_rw, rtol=1e-6, atol=1e-8)

                            y_sym = alc.fit_predict(E_sym, alc_method=cfg.alc_method, cn=cn, G_mode=cfg.G_mode)
                            y_rw  = alc.fit_predict(E_rw,  alc_method=cfg.alc_method, cn=cn, G_mode=cfg.G_mode)
                            labels_exact = np.array_equal(y_sym, y_rw)

                            # eigenvalue + gap features (from cached lap run)
                            lam_r = float(eigvals_cached[r-1]) if eigvals_cached is not None and len(eigvals_cached) >= r else np.nan
                            lam_rp1 = float(eigvals_cached[r]) if eigvals_cached is not None and len(eigvals_cached) >= (r + 1) else np.nan
                            gap_rp1 = float(lam_rp1 - lam_r) if np.isfinite(lam_r) and np.isfinite(lam_rp1) else np.nan

                            checks_rows.append({
                                "seed": seed,
                                "sigma_graph": sig,
                                "laplacian": "random_walk" if lap == "RW" else "symmetric",
                                "rank_r": r,
                                "eigvals_close": bool(eigvals_close),
                                "E_close": bool(E_close),
                                "min_cos_angle": min_cos_angle,
                                "distances_close": bool(distances_close),
                                "labels_exact": bool(labels_exact),
                                **deg_stats,
                                "lambda_r": lam_r,
                                "lambda_rp1": lam_rp1,
                                "gap_rp1": gap_rp1,
                            })

                        # Main: embedding at rank r, then ALC clustering
                        E_r, _eigvals_r = spectral_embed(
                            X=X, sigma=sig, k=n_clusters, q=r, Laplacian=lap, seed=seed
                        )
                        
                        y_pred = alc.fit_predict(E_r, cn=cn)


                        bal = cluster_balance_stats(y_pred)

                        rows.append({
                            "seed": seed,
                            "laplacian": "random_walk" if lap == "RW" else "symmetric",
                            "sigma_graph": sig,
                            "rank_r": r,
                            "ARI": adjusted_rand_score(y_true, y_pred),
                            "n_clusters": int(len(np.unique(y_pred))),
                            **bal,
                        })

                        pbar.set_postfix(seed=seed, sigma=sig, lap=lap, r=r)
                        pbar.update(1)

    df_raw = pd.DataFrame(rows)
    df_agg = (df_raw
        .groupby(["laplacian", "sigma_graph", "rank_r"], as_index=False)
        .agg(
            ARI_mean=("ARI", "mean"),
            ARI_std=("ARI", "std"),
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_std=("n_clusters", "std"),
            min_cluster_frac_mean=("min_cluster_frac", "mean"),
            min_cluster_frac_std=("min_cluster_frac", "std"),
            max_cluster_frac_mean=("max_cluster_frac", "mean"),
            max_cluster_frac_std=("max_cluster_frac", "std"),
            entropy_mean=("cluster_entropy", "mean"),
            entropy_std=("cluster_entropy", "std"),
            gini_mean=("cluster_gini", "mean"),
            gini_std=("cluster_gini", "std"),
        )
    )

    df_raw.to_csv(raw_csv, index=False)
    df_agg.to_csv(agg_csv, index=False)

    df_checks = None
    if cfg.compare_sym_rw or cfg.log_eigs:
        df_checks = pd.DataFrame(checks_rows)
        df_checks.to_csv(checks_csv, index=False)

    # -----------------------------
    # Figure: ARI_mean vs rank, faceted by sigma, line per laplacian
    # -----------------------------
    fig_path = os.path.join(cfg.figures_dir, f"{tag}_ari_vs_rank.pdf")

    sigs_sorted = sorted(df_agg["sigma_graph"].unique())
    n_sig = len(sigs_sorted)
    fig, axes = plt.subplots(1, n_sig, figsize=(5 * n_sig, 4), constrained_layout=True)

    if n_sig == 1:
        axes = [axes]

    for ax, sig in zip(axes, sigs_sorted):
        sub = df_agg[df_agg["sigma_graph"] == sig]

        for lap in sub["laplacian"].unique():
            s2 = sub[sub["laplacian"] == lap].sort_values("rank_r")
            ax.plot(s2["rank_r"], s2["ARI_mean"], marker="o", label=lap)
            ax.fill_between(
                s2["rank_r"],
                s2["ARI_mean"] - s2["ARI_std"].fillna(0.0),
                s2["ARI_mean"] + s2["ARI_std"].fillna(0.0),
                alpha=0.15
            )

        ax.set_title(f"{cfg.dataset} | sigma={sig:g}")
        ax.set_xlabel("Rank r (number of eigenvectors)")
        ax.set_ylabel("ARI (mean ± std)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {raw_csv}")
    print(f"Saved: {agg_csv}")
    if df_checks is not None:
        print(f"Saved: {checks_csv}")
    print(f"Saved: {fig_path}")

    return df_raw, df_agg, df_checks, fig_path


# %%
if __name__ == "__main__":
    # Edit this block in Spyder to run different datasets/settings.
    cfg = Cfg(
        dataset="moons",                 # "blobs" | "circles" | "moons"
        sigma_graph=[10],   # sigma grid
        laplacians=["SYM", "RW"],
        ranks=list(range(6,8)),
        seeds=list(range(1)),
        alc_method="GM",
        cn=None,                         # defaults to truth from resolve_dataset
        G_mode="corr",
        log_eigs=False,                  # set True if you want df_checks populated without compare_sym_rw
        compare_sym_rw=False,            # set True to populate df_checks with SYM-vs-RW checks
    )
    run(cfg)
