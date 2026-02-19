import numpy as np


def auto_dbscan_eps_knee(X_scaled: np.ndarray, min_samples: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    if X_scaled.shape[0] <= min_samples:
        return 0.5

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    n = k_distances.shape[0]
    x = np.arange(n, dtype=float)
    y = k_distances.astype(float)

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    line = x_norm
    diff = y_norm - line
    idx = int(np.argmax(diff))
    return float(k_distances[idx] * 5)


def auto_dbscan_eps_quantile(
    X_scaled: np.ndarray,
    min_samples: int,
    q_eps: float,
) -> float:
    from sklearn.neighbors import NearestNeighbors

    if X_scaled.shape[0] <= min_samples:
        return 0.5

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    q = float(np.clip(q_eps, 0.0, 1.0))
    return float(np.quantile(k_distances, q))
