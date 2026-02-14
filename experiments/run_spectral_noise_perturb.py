from dataclasses import dataclass
from typing import List
import os
import pandas as pd

from utils.io import ensure_dir
from utils.seed import set_global_seed

from experiments.base import run_one_condition
from experiments.common import resolve_dataset, build_methods, add_noise


# %%
@dataclass
class Cfg:
    dataset: str = "moons"
    seeds: int = 10
    noise_stds: List[float] = None
    sigmas: List[float] = None
    laplacians: List[str] = None
    methods: List[str] = None
    q: int | None = None
    output_dir: str = "results"
    csv_name: str = ""


# %%
def run(cfg: Cfg) -> pd.DataFrame:
    ensure_dir(cfg.output_dir)

    if cfg.noise_stds is None:
        cfg.noise_stds = [0.5, 1.0, 5.0]
    if cfg.sigmas is None:
        cfg.sigmas = [0.35]
    if cfg.laplacians is None:
        cfg.laplacians = ["sym", "rw"]
    if cfg.methods is None:
        cfg.methods = ["kmeans", "alc"]

    maker, data_kwargs, n_clusters = resolve_dataset(cfg.dataset)
    q = cfg.q if cfg.q is not None else n_clusters

    csv_path = os.path.join(cfg.output_dir, cfg.csv_name or f"spectral_noise_perturb_{cfg.dataset}.csv")

    for seed in range(cfg.seeds):
        set_global_seed(seed)
        X, y_true = maker(seed=seed, **data_kwargs)

        method_adapters = build_methods(cfg.methods, n_clusters=n_clusters, seed=seed)

        for noise_std in cfg.noise_stds:
            Xn = add_noise(X, std=noise_std, seed=seed)

            for lap in cfg.laplacians:
                for sigma in cfg.sigmas:
                    for method in method_adapters:
                        cond = {"noise_std": noise_std}
                        run_one_condition(
                            X=Xn,
                            y_true=y_true,
                            n_clusters=n_clusters,
                            method_adapter=method,
                            sigma=sigma,
                            laplacian_kind=lap,
                            q=q,
                            seed=seed,
                            csv_path=csv_path,
                            dataset_name=cfg.dataset,
                            experiment_name="noise_perturb",
                            condition_dict=cond,
                        )

    return pd.read_csv(csv_path)


# %%
if __name__ == "__main__":
    cfg = Cfg(dataset="moons", seeds=5, noise_stds=[0.5, 1.0], sigmas=[0.35], laplacians=["sym", "rw"])
    df = run(cfg)
    print(df.head())
