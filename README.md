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

## Repository status

The codebase is being consolidated from exploratory notebooks into reproducible Python modules and scripts suitable for a research-grade repository. Notebooks may be retained for transparency, but the intended interface is via functional `.py` code for end-to-end reproducibility.

## References

- von Luxburg, U. (2007). *A tutorial on spectral clustering*.
- Shi, J., & Malik, J. (2000). *Normalized cuts and image segmentation*.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). *On spectral clustering: Analysis and an algorithm*.
- Yelibi, L., & Gebbie, T. (2021). *Agglomerative Likelihood Clustering (ALC)*.

## Citation

If you use or build on this work, please cite the associated MSc dissertation and the references above. (A `CITATION.cff` file can be added once the dissertation citation details are finalised.)

## Licence

A licence file will be added at the repository root (e.g. MIT or BSD-3-Clause), depending on institutional and dissemination requirements.
