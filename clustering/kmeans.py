import numpy as np

class KMeansAdapter:
        """
        Custom implementation of K-Means clustering.
        Compatible with the experiment framework: exposes .fit_predict(X) and .get_params().
        """

        def __init__(self, n_clusters, random_state=0, max_iter=300):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.max_iter = max_iter
            self.centroids_ = None
            self.labels_ = None
            self.errors_ = []

        def _initialise_centroids(self, X):
            """
            Select initial centroids randomly (or deterministically via seed).
            """
            np.random.seed(self.random_state)
            n = X.shape[0]
            idx = np.random.choice(n, self.n_clusters, replace=False)
            return X[idx, :]

        def fit_predict(self, X):
            """
            Runs K-Means clustering and returns labels (same interface as sklearn).
            """
            # --- Initialise ---
            Mus = self._initialise_centroids(X)
            N, d = X.shape
            K = self.n_clusters
            errors = []

            for it in range(1, self.max_iter + 1):
                # Step 1: Compute distances
                dists = np.zeros((N, K))
                for j in range(K):
                    dists[:, j] = np.sum((X - Mus[j, :]) ** 2, axis=1)

                # Step 2: Assign clusters
                assigned_labels = np.argmin(dists, axis=1)
                errors.append(np.sum(np.min(dists, axis=1)))

                # Step 3: Update centroids
                for k in range(K):
                    cluster_points = X[assigned_labels == k]
                    if len(cluster_points) > 0:
                        Mus[k, :] = np.mean(cluster_points, axis=0)
                    else:
                        # Handle empty cluster: reassign farthest point
                        wh_rep = np.argmax(np.max(dists, axis=1))
                        Mus[k, :] = X[wh_rep, :]

                # --- Optional convergence check ---
                if it > 1 and np.allclose(errors[-1], errors[-2], atol=1e-6):
                    break

            # Store results
            self.centroids_ = Mus
            self.labels_ = assigned_labels
            self.errors_ = errors

            return assigned_labels  # compatible with rest of code

        def get_params(self):
            """
            Used by experiment logging.
            """
            return {
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "random_state": self.random_state
            }



