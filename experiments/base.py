import time
from utils.io import append_row_csv
from utils.metrics import compute_ari, compute_silhouette, eigengap
from clustering.spectral import spectral_embed

def run_one_condition(
    X, y_true, n_clusters, method_adapter, sigma, laplacian_kind, q, seed,
    csv_path, dataset_name, experiment_name, condition_dict
):
    t0 = time.time() # start time
    Y, evals = spectral_embed(X, sigma=sigma, k=n_clusters, q=q,
                              Laplacian=laplacian_kind, seed=seed)
    labels = method_adapter.fit_predict(Y) # cluster configuration output
    t1 = time.time()

    row = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "method": method_adapter.__class__.__name__.replace("Adapter","").lower(),
        "laplacian": laplacian_kind,
        "sigma": sigma,
        "q": q if q is not None else n_clusters,
        "ARI": compute_ari(y_true, labels),
        "Silhouette": compute_silhouette(Y, labels),
        "eigengap": eigengap(evals, n_clusters),
        "runtime_sec": t1 - t0,
    }
    row.update(condition_dict or {})
    append_row_csv(csv_path, row)

def run_variability (
    X, y_true, n_clusters, method_adapter, sigma, laplacian_kind, q, seed,
    csv_path, dataset_name, experiment_name, condition_dict
):
    t0 = time.time() # start time
    Y, evals = spectral_embed(X, sigma=sigma, k=n_clusters, q=q,
                              laplacian_kind=laplacian_kind, seed=seed)
    labels = method_adapter.fit_predict(Y) # cluster configuration output
    t1 = time.time()

    row = {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "seed": seed,
        "method": method_adapter.__class__.__name__.replace("Adapter","").lower(),
        "laplacian": laplacian_kind,
        "sigma": sigma,
        "q": q if q is not None else n_clusters,
        #"ARI": compute_ari(y_true, labels),
        "Silhouette": compute_silhouette(Y, labels),
        "eigengap": eigengap(evals, n_clusters),
        "runtime_sec": t1 - t0,
    }
    row.update(condition_dict or {})
    append_row_csv(csv_path, row)
