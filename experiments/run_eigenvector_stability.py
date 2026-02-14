"""
Eigenvector stability experiment (clean data, resampling-based overlap matrices).

Runs for: blobs, circles, moons
- generates clean dataset (no noise)
- for each resample percentage p, draws two independent subsamples of size floor(pN)
- computes leading eigenvectors U1, U2
- stores overlap matrix O = U2^T U1 (optionally absolute)
- saves O for each p and saves a 2x3 heatmap panel figure

Designed to run end-to-end in Spyder or from the terminal.
"""

from dataclasses import dataclass
from typing import List, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

from utils.io import ensure_dir
from utils.seed import set_global_seed
from experiments.common import resolve_dataset  # uses blobs/circles/moons -> generator
from clustering.spectral import dist_mat, KM_spec_decom


# %%
@dataclass
class Cfg:
    # Core experiment settings
    dataset: str = "blobs"  # blobs | circles | moons
    seed: int = 42

    sample_perc: List[float] = None  # must contain 1.0 if you want anchored colour scale
    sigma: float = 10.0
    laplacian: str = "SYM"           # match what your KM_spec_decom expects ("SYM", "RW", etc.)
    lead_vecs_dim: int = 20

    use_abs_overlap: bool = True

    # Output locations
    results_dir: str = "results/eigenvector_stability"
    figures_dir: str = "figures/eigenvector_stability"
    save_matrices_as: str = "npy"    # "npy" or "npz"

    def __post_init__(self):
        if self.sample_perc is None:
            self.sample_perc = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]


# %%
def compute_leading_evecs(X: np.ndarray, sigma: float, laplacian: str, k: int) -> np.ndarray:
    """
    Build similarity matrix from X, compute Laplacian eigendecomposition, return leading eigenvectors U.
    """
    sim = dist_mat(X)                  # expected: returns pairwise distance matrix (your implementation)
    I = np.eye(sim.shape[0])

    eig_vals, eig_vecs, _A = KM_spec_decom(
        sim_mat=sim,
        sigma=sigma,
        Laplacian=laplacian,
        I_mat=I
    )
    return eig_vecs[:, :k]


# %%
def run(cfg: Cfg):
    """
    Returns:
        heat_map_dict: dict[p] -> overlap matrix O (k x k)
        fig_path: saved figure path
        mat_paths: dict[p] -> saved matrix path
    """
    set_global_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    ensure_dir(cfg.results_dir)
    ensure_dir(cfg.figures_dir)

    # -----------------------
    # 1) Generate clean dataset
    # -----------------------
    maker, data_kwargs, _n_clusters = resolve_dataset(cfg.dataset)
    X, _y_true = maker(seed=cfg.seed, **data_kwargs)

    N = X.shape[0]
    indices = np.arange(N)

    # -----------------------
    # 2) Compute overlap matrices
    # -----------------------
    heat_map_dict: Dict[float, np.ndarray] = {}
    mat_paths: Dict[float, str] = {}

    # Save naming
    tag = f"{cfg.dataset}_sigma{cfg.sigma:g}_{cfg.laplacian}_k{cfg.lead_vecs_dim}_seed{cfg.seed}"
    base_mat_dir = os.path.join(cfg.results_dir, tag)
    ensure_dir(base_mat_dir)

    for p in cfg.sample_perc:
        p = float(p)
        n_samp = int(N * p)
        if n_samp < cfg.lead_vecs_dim:
            raise ValueError(
                f"Resample size {n_samp} is smaller than lead_vecs_dim={cfg.lead_vecs_dim}. "
                f"Increase p or reduce lead_vecs_dim."
            )

        sampled_1 = rng.choice(indices, size=n_samp, replace=False)
        sampled_2 = rng.choice(indices, size=n_samp, replace=False)

        X1 = X[sampled_1, :]
        X2 = X[sampled_2, :]

        U1 = compute_leading_evecs(X1, sigma=cfg.sigma, laplacian=cfg.laplacian, k=cfg.lead_vecs_dim)
        U2 = compute_leading_evecs(X2, sigma=cfg.sigma, laplacian=cfg.laplacian, k=cfg.lead_vecs_dim)

        O = U2.T @ U1
        if cfg.use_abs_overlap:
            O = np.abs(O)

        heat_map_dict[p] = O

        # Persist each overlap matrix
        fname = f"overlap_p{int(p*100):02d}.npy" if cfg.save_matrices_as == "npy" else f"overlap_p{int(p*100):02d}.npz"
        out_path = os.path.join(base_mat_dir, fname)

        if cfg.save_matrices_as == "npy":
            np.save(out_path, O)
        else:
            np.savez_compressed(out_path, O=O, p=p, sigma=cfg.sigma, laplacian=cfg.laplacian, k=cfg.lead_vecs_dim, seed=cfg.seed)

        mat_paths[p] = out_path
        print(f"{int(p*100)}% done -> {out_path}")

    # Also save the whole dict as a single npz for convenience
    dict_path = os.path.join(base_mat_dir, "overlaps_all.npz")
    np.savez_compressed(
        dict_path,
        **{f"p_{int(p*100):02d}": heat_map_dict[p] for p in heat_map_dict},
        sample_perc=np.array(cfg.sample_perc, dtype=float),
        sigma=cfg.sigma,
        laplacian=cfg.laplacian,
        k=cfg.lead_vecs_dim,
        seed=cfg.seed,
        use_abs_overlap=cfg.use_abs_overlap,
        dataset=cfg.dataset,
    )
    print(f"Saved combined overlaps -> {dict_path}")

    # -----------------------
    # 3) Plot 2x3 panel (anchored colour scale to p=1.0)
    # -----------------------
    if 1.0 not in heat_map_dict:
        raise ValueError("sample_perc must include 1.0 to anchor the shared colour scale.")

    full_O = heat_map_dict[1.0]
    vmin = float(full_O.min())
    vmax = float(full_O.max())
    if cfg.use_abs_overlap:
        vmin = 0.0
        vmax = float(full_O.max())

    # Your diverging style
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["darkblue", "white", "darkred"])

    titles = {
        1.0: "Full dataset",
        0.9: "90% resample",
        0.7: "70% resample",
        0.5: "50% resample",
        0.3: "30% resample",
        0.1: "10% resample",
    }

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    mappable = None
    k = cfg.lead_vecs_dim

    for ax, p in zip(axs.flatten(), [float(x) for x in cfg.sample_perc]):
        O = heat_map_dict[p]
        hm = sns.heatmap(
            O,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=custom_cmap,
            square=True,
            cbar=False
        )
        ax.set_xticks(np.arange(k) + 0.5)
        ax.set_yticks(np.arange(k) + 0.5)
        ax.set_xticklabels(np.arange(1, k + 1))
        ax.set_yticklabels(np.arange(1, k + 1))
        ax.set_title(titles.get(p, f"{int(p*100)}% resample"), fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        mappable = hm.collections[0]

    fig.supxlabel(f"Leading eigenvectors (1–{k})", fontsize=14)
    fig.supylabel(f"Leading eigenvectors (1–{k})", fontsize=14)

    cbar = fig.colorbar(mappable, ax=axs, shrink=0.9, pad=0.02)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    if cfg.use_abs_overlap:
        cbar.set_label(r"$|u_i^\top u_j|$", fontsize=14)
    else:
        cbar.set_label(r"$u_i^\top u_j$", fontsize=14)

    fig_name = f"eigenvector_stability_panel_{tag}.pdf"
    fig_path = os.path.join(cfg.figures_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {fig_path}")

    return heat_map_dict, fig_path, mat_paths


# %%
if __name__ == "__main__":
    # Edit these defaults in Spyder (or run from terminal)
    cfg = Cfg(dataset="blobs", sigma=10.0, laplacian="SYM", lead_vecs_dim=20, seed=42)
    overlaps, fig_path, mat_paths = run(cfg)
