"""
Smoke test runner: executes a minimal end-to-end run across all primary datasets.

- Raw ALC on raw data (1 seed)
- Spectral baseline (1 seed, 1 sigma, 1 laplacian; KMeans + ALC)
- Spectral + noise perturbation (1 seed, 1 sigma, 1 laplacian; KMeans + ALC)

Outputs CSVs to results/.
Designed to be run top-to-bottom in Spyder.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import List
import pandas as pd

from experiments.run_raw_alc import Cfg as RawCfg, run as run_raw
from experiments.run_spectral_baseline_sigma import Cfg as SpecCfg, run as run_spec
from experiments.run_spectral_noise_perturb import Cfg as NoiseCfg, run as run_noise


# %%
@dataclass
class SmokeCfg:
    datasets: List[str] = None       # ["blobs", "circles", "moons"]
    seeds: int = 1                   # minimal run
    sigma: float = 0.35
    laplacian: str = "sym"           # or "rw"
    noise_std: float = 0.5
    methods: List[str] = None        # ["kmeans", "alc"]

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["blobs", "circles", "moons"]
        if self.methods is None:
            self.methods = ["kmeans", "alc"]


# %%
def run_smoke(cfg: SmokeCfg):
    all_results = []

    for ds in cfg.datasets:
        print(f"\n=== SMOKE TEST: {ds} ===")

        # 1) Raw ALC
        df_raw = run_raw(RawCfg(dataset=ds, seeds=cfg.seeds, output_dir="results",
                                csv_name=f"smoke_raw_alc_{ds}.csv"))
        print(f"[OK] raw_alc: {len(df_raw)} rows -> results/smoke_raw_alc_{ds}.csv")
        all_results.append(df_raw)

        # 2) Spectral baseline (single sigma, single laplacian)
        df_spec = run_spec(SpecCfg(dataset=ds, seeds=cfg.seeds, output_dir="results",
                                   sigmas=[cfg.sigma], laplacians=[cfg.laplacian],
                                   methods=cfg.methods,
                                   csv_name=f"smoke_spectral_baseline_{ds}.csv"))
        print(f"[OK] spectral_baseline: {len(df_spec)} rows -> results/smoke_spectral_baseline_{ds}.csv")
        all_results.append(df_spec)

        # 3) Spectral + noise perturbation (single noise_std, single sigma, single laplacian)
        df_noise = run_noise(NoiseCfg(dataset=ds, seeds=cfg.seeds, output_dir="results",
                                      noise_stds=[cfg.noise_std], sigmas=[cfg.sigma],
                                      laplacians=[cfg.laplacian], methods=cfg.methods,
                                      csv_name=f"smoke_spectral_noise_{ds}.csv"))
        print(f"[OK] spectral_noise: {len(df_noise)} rows -> results/smoke_spectral_noise_{ds}.csv")
        all_results.append(df_noise)

    df_all = pd.concat(all_results, ignore_index=True)
    print("\n=== SMOKE TEST COMPLETE ===")
    print(df_all.groupby(["dataset", "experiment", "method"]).size())

    return df_all


# %%
if __name__ == "__main__":
    cfg = SmokeCfg(seeds=1, sigma=0.35, laplacian="sym", noise_std=0.5)
    df = run_smoke(cfg)
    print("\nHead:")
    print(df.head())
