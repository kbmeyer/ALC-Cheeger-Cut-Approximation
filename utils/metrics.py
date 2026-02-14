import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

def compute_ari(y_true, y_pred):
    return float(adjusted_rand_score(y_true, y_pred))

def compute_silhouette(X_embedded, labels):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) == len(labels):
        return float("nan")
    try:
        return float(silhouette_score(X_embedded, labels))
    except Exception:
        return float("nan")

def eigengap(vals, k):
    if k <= 0 or k >= len(vals):
        return float("nan")
    return float(vals[k] - vals[k-1])
