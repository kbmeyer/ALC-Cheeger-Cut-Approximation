from dataclasses import dataclass
import os
import pandas as pd

from utils.io import ensure_dir, append_row_csv
from utils.seed import set_global_seed
from utils.metrics import compute_ari, compute_silhouette

from clustering.alc import ALCAdapter
from experiments.common import resolve_dataset


# %%
@dataclass
class Cfg:
    dataset: str = "moons"      # blobs | circles | moons
    seeds: int = 10
    output_dir: str = "results"
    csv_name: str = ""          # leave blank to auto-name


# %%
def run(cfg: Cfg) -> pd.DataFrame:
    ensure_dir(cfg.output_dir)

    maker, data_kwargs, n_clusters = resolve_dataset(cfg.dataset)

    csv_path = os.path.join(cfg.output_dir, cfg.csv_name or f"raw_alc_{cfg.dataset}.csv")

    for seed in range(cfg.seeds):
        set_global_seed(seed)
        X, y_true = maker(seed=seed, **data_kwargs)

        t0 = pd.Timestamp.utcnow()
        alc = ALCAdapter()

        # Your adapter currently expects cn; for raw baseline we use the known q
        y_pred = alc.fit_predict(X, cn=n_clusters)

        runtime_sec = (pd.Timestamp.utcnow() - t0).total_seconds()

        row = {
            "dataset": cfg.dataset,
            "experiment": "raw_alc",
            "seed": seed,
            "method": "alc",
            "q": n_clusters,
            "ARI": compute_ari(y_true, y_pred),
            "Silhouette": compute_silhouette(X, y_pred),
            "runtime_sec": runtime_sec,
        }
        append_row_csv(csv_path, row)

    return pd.read_csv(csv_path)


# %%
if __name__ == "__main__":
    cfg = Cfg(dataset="moons", seeds=5)
    df = run(cfg)
    print(df.head())
