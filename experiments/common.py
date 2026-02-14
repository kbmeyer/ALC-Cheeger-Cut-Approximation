from typing import Callable, Dict, Tuple, Any, List
import numpy as np

from data.generators import make_blobs_convex, make_rings, make_two_moons
from clustering.kmeans import KMeansAdapter
from clustering.alc import ALCAdapter


def resolve_dataset(dataset: str) -> Tuple[Callable[..., Tuple[np.ndarray, np.ndarray]], Dict[str, Any], int]:
    """
    Returns:
        maker(seed=..., **data_kwargs) -> (X, y_true)
        data_kwargs
        n_clusters (ground-truth; used as default q and for KMeans)
    """
    d = dataset.lower().strip()

    if d == "blobs":
        data_kwargs = dict(N=1200, num_clusters=7, n_features=5000)
        def maker(seed: int, **kw):
            return make_blobs_convex(seed=seed, **kw)
        return maker, data_kwargs, 7

    if d in {"circles", "rings"}:
        # User-facing name is circles; underlying generator is currently make_rings
        data_kwargs = dict(n_points=1200, n_circles=3, noise_std=0.1)
        def maker(seed: int, **kw):
            return make_rings(seed=seed, **kw)
        return maker, data_kwargs, 3

    if d in {"moons", "two_moons"}:
        data_kwargs = dict(n_samples=1200, noise=0.05)
        def maker(seed: int, **kw):
            return make_two_moons(seed=seed, **kw)
        return maker, data_kwargs, 2

    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: blobs, circles, moons.")


def build_methods(method_names: List[str], n_clusters: int, seed: int):
    """
    Build fresh method instances per seed (important for KMeans random_state).
    """
    methods = []
    for name in method_names:
        n = name.lower().strip()
        if n == "kmeans":
            methods.append(KMeansAdapter(n_clusters=n_clusters, random_state=seed))
        elif n == "alc":
            methods.append(ALCAdapter())
        else:
            raise ValueError(f"Unknown method '{name}'. Choose from: alc, kmeans.")
    return methods


def add_noise(X: np.ndarray, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, std, size=X.shape)
