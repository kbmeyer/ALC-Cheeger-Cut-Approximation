# ALC for Cheeger-Style Spectral Clustering: Eigenvector Rank, Laplacians, and Robustness

This repository contains the code accompanying an MSc dissertation investigating the **Cheeger cut** problem via **spectral clustering**, with a particular focus on embedding **Agglomerative Likelihood Clustering (ALC)** (Yelibi & Gebbie, 2021) into the spectral clustering framework to study (i) the **minimum number of Laplacian eigenvectors** required for accurate low-dimensional representation and clustering, and (ii) the **robustness** of ALC under different graph constructions, Laplacian choices, and perturbation regimes.

## Overview

Spectral clustering constructs a similarity graph over data, computes a graph Laplacian, and uses the leading eigenvectors to obtain a low-dimensional embedding on which clustering is performed. In many settings, the number of clusters present in the data is closely aligned with the minimum number of eigenvectors needed to separate those clusters. However, standard pipelines typically require the number of clusters to be selected *a priori*.

This dissertation explores whether **ALC**—which does not require specifying the number of clusters in advance—can be used within the spectral pipeline to:
- recover a **Cheeger-style separator** by interpreting the number of clusters inferred by ALC as an approximation to the underlying cut structure; and
- provide empirical clarity on **how many eigenvectors are sufficient** for stable, accurate clustering.

A second clustering algorithm (**K-Means**) is embedded into the same spectral framework as a baseline comparator.

## Methodological components

The main experimental pipeline consists of:
1. **Similarity graph construction** using a **Gaussian kernel** with bandwidth parameter \(\sigma\).
2. **Graph Laplacian construction** using multiple variants (as surveyed in von Luxburg, 2007).
3. **Spectral decomposition** and formation of a low-dimensional **spectral embedding** from the leading eigenvectors.
4. **Clustering in the embedding space** using:
   - **ALC** (no pre-specified number of clusters), and
   - **K-Means** (baseline; requires specifying the number of clusters).
5. **Evaluation and analysis**, including the impact of Laplacian choice on:
   - clustering performance,
   - stability of eigenvectors across regimes, and
   - the recoverability of Cheeger-style partitions.

## Experiments (high-level)

The experimental programme assesses:
- performance on **raw data** versus **spectral embeddings**;
- sensitivity to the Gaussian kernel bandwidth \(\sigma\);
- differences between **Laplacian constructions**;
- robustness under perturbations/noise and stability of the resulting eigenvectors;
- the effect of **rank specification** (number of eigenvectors used) on clustering outcomes.

## Repository purpose and structure

This repository contains the minimal Python code required to reproduce the experiments, figures, and summary metrics reported in the accompanying mini-dissertation. The codebase is organised to separate (i) clustering methods, (ii) synthetic data generation, (iii) experiment runners, and (iv) shared utilities.

### Directory overview

- **`clustering/`**  
  Core clustering implementations used throughout the dissertation:
  - `alc.py` — Agglomerative Likelihood Clustering (ALC) implementation.
  - `kmeans.py` — K-means baseline.
  - `spectral.py` — Spectral clustering routines (including Laplacian construction / embedding steps as used in the experiments).

- **`data/`**  
  Synthetic dataset construction:
  - `generators.py` — dataset generators used to produce the benchmark datasets referenced in the experimental chapters.

- **`experiments/`**  
  Reproducible experiment entry points and shared experiment scaffolding:
  - `base.py`, `common.py` — shared configuration, runners, and helper logic used across experiments.
  - `run_spectral_baseline_sigma.py` — baseline spectral clustering across kernel bandwidths.
  - `run_spectral_noise_perturb.py` — noise-perturbation experiments.
  - `run_spectral_rank_proxy.py` — rank / embedding-dimension experiments (proxy/overspecification regime).
  - `run_eigenvector_stability.py` — eigenvector stability and alignment experiments.
  - `run_raw_alc.py` — ALC runs without spectral preprocessing (where applicable).
  - `run_smoke_test_all.py` — convenience script to validate the core experiments (baseline and noise perturbation) pipeline end-to-end.

- **`utils/`**  
  Shared utilities for reproducibility and evaluation:
  - `io.py` — loading/saving results and artefacts.
  - `metrics.py` — evaluation metrics (e.g., ARI, silhouette, and related summaries).
  - `seed.py` — seed control and reproducibility helpers.

### Reproducibility note

The experiment scripts under `experiments/` are intended to be the primary entry points for reproducing dissertation results. Each script executes a defined experimental loop (e.g., over seeds, noise levels, Laplacian variants, or bandwidth parameters) and outputs the corresponding results used in downstream analysis and plotting.


## References

- von Luxburg, U. (2007). *A tutorial on spectral clustering*.
- Shi, J., & Malik, J. (2000). *Normalized cuts and image segmentation*.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). *On spectral clustering: Analysis and an algorithm*.
- Yelibi, L., & Gebbie, T. (2021). *Agglomerative Likelihood Clustering (ALC)*.

