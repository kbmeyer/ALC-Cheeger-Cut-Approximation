from dataclasses import dataclass
from typing import List
import os
import pandas as pd

from utils.io import ensure_dir
from utils.seed import set_global_seed

from experiments.base import run_one_condition
from experiments.common import resolve_dataset, build_methods


# %%
@dataclass
class Cfg:
    dataset: str = "moons"                 # blobs | circles | moons
    seeds: int = 10
    sigmas: List[float] = None
    laplacians: List[str] = None           # e.g. ["sym", "rw"]
    methods: List[str] = None              # ["kmeans", "alc"]
    q: int | None = None                   # embedding rank; None -> n_clusters
    output_dir: str = "results"
    csv_name: str = ""


# %%
def run(cfg: Cfg) -> pd.DataFrame:
    ensure_dir(cfg.output_dir)

    if cfg.sigmas is None:
        cfg.sigmas = [0.25, 0.35, 0.45]
    if cfg.laplacians is None:
        cfg.laplacians = ["sym", "rw"]
    if cfg.methods is None:
        cfg.methods = ["kmeans", "alc"]

    maker, data_kwargs, n_clusters = resolve_dataset(cfg.dataset)
    q = cfg.q if cfg.q is not None else n_clusters

    csv_path = os.path.join(cfg.output_dir, cfg.csv_name or f"spectral_baseline_sigma_{cfg.dataset}.csv")

    for seed in range(cfg.seeds):
        set_global_seed(seed)
        X, y_true = maker(seed=seed, **data_kwargs)

        # Build methods PER SEED so KMeans random_state varies correctly
        method_adapters = build_methods(cfg.methods, n_clusters=n_clusters, seed=seed)

        for lap in cfg.laplacians:
            for sigma in cfg.sigmas:
                for method in method_adapters:
                    cond = {"baseline_sigma": sigma}
                    run_one_condition(
                        X=X,
                        y_true=y_true,
                        n_clusters=n_clusters,
                        method_adapter=method,
                        sigma=sigma,
                        laplacian_kind=lap,
                        q=q,
                        seed=seed,
                        csv_path=csv_path,
                        dataset_name=cfg.dataset,
                        experiment_name="baseline_sigma",
                        condition_dict=cond,
                    )

    return pd.read_csv(csv_path)


# %%
if __name__ == "__main__":
    cfg = Cfg(dataset="moons", seeds=5, sigmas=[0.25, 0.35], laplacians=["sym", "rw"], methods=["kmeans", "alc"])
    df = run(cfg)
    print(df.head())
