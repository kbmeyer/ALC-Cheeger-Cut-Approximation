from sklearn.datasets import make_blobs, make_moons
import numpy as np

# ---------------- Rings generator ----------------
def make_rings(n_points=1200, n_circles=3, noise_std=0.1, seed=0):
    """
    Generate concentric ring (circle) data with controllable random seed.

    Parameters
    ----------
    n_points : int
        Total number of points.
    n_circles : int
        Number of rings.
    noise_std : float
        Standard deviation of radial noise.
    seed : int
        Random seed for reproducibility / Monte Carlo resampling.

    Returns
    -------
    xy_coordinates : ndarray of shape (n_points, 2)
    ring_labels : ndarray of shape (n_points,)
    """
    rng = np.random.default_rng(seed)

    # Assign each point to a ring label
    ring_labels = rng.choice(np.arange(1, n_circles + 1), n_points, replace=True)

    # Radial position with small Gaussian noise
    r = 2 * ring_labels + rng.normal(0, noise_std, n_points)

    # Angular coordinate (uniform on [0, 2π))
    theta = rng.uniform(0, 2 * np.pi, n_points)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    xy_coordinates = np.column_stack((x, y))

    return xy_coordinates, ring_labels


# ---------------- High-dimensional convex blobs ----------------
def make_blobs_convex(N=1200, num_clusters=7, n_features=5000, seed=0):
    """
    Generate high-dimensional Gaussian blobs with a fixed seed.

    Parameters
    ----------
    N : int
        Total number of samples.
    num_clusters : int
        Number of clusters.
    n_features : int
        Dimensionality of feature space.
    seed : int
        Random seed for reproducibility / Monte Carlo resampling.

    Returns
    -------
    blobs : ndarray of shape (N, n_features)
    labels : ndarray of shape (N,)
    """
    blobs, labels = make_blobs(
        n_samples=N,
        n_features=n_features,
        centers=num_clusters,
        shuffle=False,
        random_state=seed
    )
    return blobs, labels

def make_two_moons(n_samples=1200, noise=0.05, seed=0):
    xy_coordinates, moon_labels = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    moon_labels = moon_labels + 1
    return xy_coordinates, moon_labels


