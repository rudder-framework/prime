"""
Fleet baseline: pooled early-life eigenstructure.

Take the first N% of each cohort's lifetime, pool all signal vectors,
compute eigendecomposition. This is the "healthy" reference.

Observation geometry (per-cycle centroid distance and PC1 projection)
is computed relative to this baseline.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_fleet_baseline(
    cohort_matrices: Dict[str, np.ndarray],
    early_fraction: float = 0.2,
    min_windows: int = 5,
) -> Dict[str, Any]:
    """
    Compute fleet baseline from early-life windows of multiple cohorts.

    Parameters
    ----------
    cohort_matrices : dict
        {cohort_id: np.ndarray(n_windows, n_features)} — signal vector
        matrices per cohort over time.
    early_fraction : float
        Fraction of each cohort's life to use as baseline.
    min_windows : int
        Minimum windows per cohort to include.

    Returns
    -------
    dict with:
        centroid : np.ndarray — fleet baseline centroid
        eigenvalues : np.ndarray
        eigenvectors : np.ndarray
        effective_dim : float
        n_cohorts : int
        n_pooled_windows : int
    """
    pooled = []

    for cohort_id, matrix in cohort_matrices.items():
        n = len(matrix)
        if n < min_windows:
            continue
        early_n = max(min_windows, int(n * early_fraction))
        pooled.append(matrix[:early_n])

    if not pooled:
        return _empty_baseline()

    pooled_matrix = np.vstack(pooled)
    n_pooled, n_features = pooled_matrix.shape

    if n_pooled < 3:
        return _empty_baseline()

    # Remove NaN rows
    valid_mask = np.isfinite(pooled_matrix).all(axis=1)
    pooled_clean = pooled_matrix[valid_mask]

    if len(pooled_clean) < 3:
        return _empty_baseline()

    centroid = np.mean(pooled_clean, axis=0)
    centered = pooled_clean - centroid

    # Covariance and eigendecomposition
    cov = np.cov(centered.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return _empty_baseline()

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-30)

    # Effective dimension
    p = eigenvalues / eigenvalues.sum()
    eff_dim = float(np.exp(-np.sum(p * np.log(p + 1e-30))))

    return {
        'centroid': centroid,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'effective_dim': eff_dim,
        'n_cohorts': len(pooled),
        'n_pooled_windows': len(pooled_clean),
    }


def compute_observation_departure(
    observation: np.ndarray,
    baseline: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute departure of a single observation from fleet baseline.

    Parameters
    ----------
    observation : np.ndarray
        Feature vector (1D) for one observation window.
    baseline : dict
        Output from compute_fleet_baseline.

    Returns
    -------
    dict with centroid_distance, pc1_projection.
    """
    obs = np.asarray(observation, dtype=np.float64).flatten()
    centroid = baseline['centroid']
    eigenvectors = baseline['eigenvectors']

    if not np.isfinite(obs).all():
        return {'centroid_distance': np.nan, 'pc1_projection': np.nan}

    dist = float(np.linalg.norm(obs - centroid))

    # Project onto PC1
    pc1 = eigenvectors[:, 0]
    centered = obs - centroid
    proj = float(np.dot(centered, pc1))

    return {
        'centroid_distance': dist,
        'pc1_projection': proj,
    }


def _empty_baseline() -> Dict[str, Any]:
    return {
        'centroid': np.array([]),
        'eigenvalues': np.array([]),
        'eigenvectors': np.array([]),
        'effective_dim': 0.0,
        'n_cohorts': 0,
        'n_pooled_windows': 0,
    }
