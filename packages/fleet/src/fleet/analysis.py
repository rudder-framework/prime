"""
Fleet-level analysis: eigendecomp, pairwise, velocity across cohorts.

Cohort centroids become the "signals" and we apply identical math
at the fleet scale. This is the recursive principle in action.
"""

import numpy as np
from typing import Dict, Any, List


def compute_fleet_eigendecomp(
    cohort_centroids: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Eigendecomposition of fleet: cohorts as rows, features as columns.

    Parameters
    ----------
    cohort_centroids : dict
        {cohort_id: np.ndarray(n_features)} — centroid per cohort
        at a single time point.

    Returns
    -------
    dict with eigenvalues, effective_dim, fleet_centroid, etc.
    """
    cohort_ids = list(cohort_centroids.keys())
    if len(cohort_ids) < 2:
        return _empty_fleet_eigen()

    matrix = np.array([cohort_centroids[c] for c in cohort_ids])
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    cohort_ids = [c for c, v in zip(cohort_ids, valid) if v]

    if len(matrix) < 2:
        return _empty_fleet_eigen()

    centroid = np.mean(matrix, axis=0)
    centered = matrix - centroid

    cov = np.cov(centered.T) if centered.shape[1] > 1 else np.array([[np.var(centered)]])
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return _empty_fleet_eigen()

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[idx], 1e-30)
    eigenvectors = eigenvectors[:, idx]

    p = eigenvalues / eigenvalues.sum()
    eff_dim = float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'effective_dim': eff_dim,
        'fleet_centroid': centroid,
        'cohort_ids': cohort_ids,
        'n_cohorts': len(cohort_ids),
    }


def compute_fleet_pairwise(
    cohort_centroids: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Pairwise metrics between all cohort centroids.

    Returns
    -------
    list of dict — one per pair (C(N,2) total).
    """
    cohort_ids = sorted(cohort_centroids.keys())
    results = []

    for i in range(len(cohort_ids)):
        for j in range(i + 1, len(cohort_ids)):
            a = np.asarray(cohort_centroids[cohort_ids[i]], dtype=np.float64)
            b = np.asarray(cohort_centroids[cohort_ids[j]], dtype=np.float64)

            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 2:
                continue

            a_v, b_v = a[valid], b[valid]

            distance = float(np.linalg.norm(a_v - b_v))

            # Cosine similarity
            norm_a, norm_b = np.linalg.norm(a_v), np.linalg.norm(b_v)
            if norm_a > 1e-12 and norm_b > 1e-12:
                cos_sim = float(np.dot(a_v, b_v) / (norm_a * norm_b))
            else:
                cos_sim = 0.0

            # Correlation
            if len(a_v) > 2:
                corr = float(np.corrcoef(a_v, b_v)[0, 1])
            else:
                corr = np.nan

            results.append({
                'cohort_a': cohort_ids[i],
                'cohort_b': cohort_ids[j],
                'distance': distance,
                'cosine_similarity': cos_sim,
                'correlation': corr,
            })

    return results


def compute_fleet_velocity(
    cohort_centroid_series: Dict[str, List[np.ndarray]],
) -> List[Dict[str, Any]]:
    """
    Velocity of fleet centroid over time.

    Parameters
    ----------
    cohort_centroid_series : dict
        {cohort_id: [centroid_t0, centroid_t1, ...]} — centroids over windows.

    Returns
    -------
    list of dict — fleet speed and direction per timestep.
    """
    # Average across cohorts at each timestep
    cohort_ids = list(cohort_centroid_series.keys())
    if not cohort_ids:
        return []

    n_windows = min(len(v) for v in cohort_centroid_series.values())
    if n_windows < 3:
        return []

    fleet_trajectory = []
    for t in range(n_windows):
        vectors = [cohort_centroid_series[c][t] for c in cohort_ids
                    if t < len(cohort_centroid_series[c])]
        fleet_trajectory.append(np.mean(vectors, axis=0))

    fleet_trajectory = np.array(fleet_trajectory)

    # Velocity
    v = np.diff(fleet_trajectory, axis=0)
    speed = np.linalg.norm(v, axis=1)

    results = []
    for i in range(len(v)):
        results.append({
            'I': i + 1,
            'fleet_speed': float(speed[i]),
        })

    return results


def _empty_fleet_eigen() -> Dict[str, Any]:
    return {
        'eigenvalues': np.array([]),
        'eigenvectors': np.array([]),
        'effective_dim': 0.0,
        'fleet_centroid': np.array([]),
        'cohort_ids': [],
        'n_cohorts': 0,
    }
