"""
Persistent homology from point clouds.

Uses Vietoris-Rips filtration on distance matrix.
Delegates to giotto-tda or ripser if available, fallback to
distance-matrix-based Betti-0 computation.

Output: persistence pairs (birth, death, dimension) + summary stats.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.spatial.distance import pdist, squareform


def compute_persistence(
    point_cloud: np.ndarray,
    max_dim: int = 1,
    max_points: int = 500,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute persistent homology of a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        (n_points, n_features) — embedded state space trajectory.
    max_dim : int
        Maximum homology dimension (0=components, 1=loops, 2=voids).
    max_points : int
        Subsample if exceeding this (O(n³) complexity).

    Returns
    -------
    dict with:
        persistence_pairs : list of (birth, death, dim) tuples
        betti_0 : int — components at median threshold
        betti_1 : int — loops at median threshold
        total_persistence_0 : float — sum(death - birth) for dim 0
        total_persistence_1 : float — sum(death - birth) for dim 1
        max_persistence_0 : float
        max_persistence_1 : float
        n_points : int
    """
    point_cloud = np.asarray(point_cloud, dtype=np.float64)

    if point_cloud.ndim == 1:
        point_cloud = point_cloud.reshape(-1, 1)

    n = len(point_cloud)
    if n < 3:
        return _empty_persistence()

    # Subsample if needed
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        idx.sort()
        point_cloud = point_cloud[idx]
        n = max_points

    # Try ripser first
    try:
        import ripser
        result = ripser.ripser(point_cloud, maxdim=max_dim)
        pairs = []
        for dim, dgm in enumerate(result['dgms']):
            for birth, death in dgm:
                if np.isfinite(death):
                    pairs.append((float(birth), float(death), dim))
        return _summarize_persistence(pairs, n)
    except ImportError:
        pass

    # Fallback: distance-matrix based Betti-0
    return _betti0_fallback(point_cloud, n)


def _betti0_fallback(point_cloud: np.ndarray, n: int) -> Dict[str, Any]:
    """Compute Betti-0 from distance matrix using union-find."""
    dists = pdist(point_cloud)
    dist_sq = squareform(dists)

    # Sort edges by distance
    triu_idx = np.triu_indices(n, k=1)
    edge_dists = dist_sq[triu_idx]
    sort_idx = np.argsort(edge_dists)

    # Union-find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False

    pairs = []
    birth_times = {i: 0.0 for i in range(n)}  # all born at distance 0

    for idx in sort_idx:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        d = float(edge_dists[idx])
        if union(i, j):
            # A component dies (merged)
            pairs.append((0.0, d, 0))

    return _summarize_persistence(pairs, n)


def _summarize_persistence(
    pairs: List[Tuple[float, float, int]],
    n_points: int,
) -> Dict[str, Any]:
    """Summarize persistence pairs."""
    result = {
        'persistence_pairs': pairs,
        'n_points': n_points,
    }

    for dim in range(2):
        dim_pairs = [(b, d) for b, d, dm in pairs if dm == dim]
        lifetimes = [d - b for b, d in dim_pairs]

        result[f'betti_{dim}'] = len(dim_pairs)
        result[f'total_persistence_{dim}'] = float(sum(lifetimes)) if lifetimes else 0.0
        result[f'max_persistence_{dim}'] = float(max(lifetimes)) if lifetimes else 0.0
        result[f'mean_persistence_{dim}'] = float(np.mean(lifetimes)) if lifetimes else 0.0

    return result


def betti_numbers_at_threshold(
    point_cloud: np.ndarray,
    threshold: float,
) -> Dict[str, int]:
    """
    Count connected components at a specific distance threshold.

    Parameters
    ----------
    point_cloud : np.ndarray
        (n_points, n_features).
    threshold : float
        Distance threshold for connectivity.

    Returns
    -------
    dict with betti_0 count.
    """
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    n = len(point_cloud)
    if n < 2:
        return {'betti_0': n}

    dists = squareform(pdist(point_cloud))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    components = len(set(find(i) for i in range(n)))
    return {'betti_0': components}


def _empty_persistence() -> Dict[str, Any]:
    return {
        'persistence_pairs': [],
        'n_points': 0,
        'betti_0': 0, 'betti_1': 0,
        'total_persistence_0': 0.0, 'total_persistence_1': 0.0,
        'max_persistence_0': 0.0, 'max_persistence_1': 0.0,
        'mean_persistence_0': 0.0, 'mean_persistence_1': 0.0,
    }
