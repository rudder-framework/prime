"""
System Vector
=============
Aggregates cohort-level vectors into system-level vectors.

Same math as cohort, one level up:
    Cohort: centroid across signals
    System: centroid across cohorts

Usage:
    from vector.system import compute_system

    row = compute_system(
        cohort_matrix=matrix,  # (n_cohorts, n_features)
        window_index=0,
        feature_names=[...],
    )
"""

import numpy as np
from typing import Dict, Any, List


def compute_system(
    cohort_matrix: np.ndarray,
    window_index: int,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Compute system vector from cohort matrix at one window.

    Args:
        cohort_matrix: (n_cohorts, n_features) array.
        window_index: Which window this is.
        feature_names: Names of the feature columns.

    Returns:
        Dict with:
            window_index,
            system_centroid_{feature} for each feature,
            system_dispersion_mean, system_dispersion_max, system_dispersion_std,
            n_cohorts, n_active_features
    """
    n_cohorts, n_features = cohort_matrix.shape

    row = {
        'window_index': window_index,
        'n_cohorts': n_cohorts,
    }

    if n_cohorts == 0:
        for name in feature_names:
            row[f'system_centroid_{name}'] = float('nan')
        row['system_dispersion_mean'] = float('nan')
        row['system_dispersion_max'] = float('nan')
        row['system_dispersion_std'] = float('nan')
        row['n_active_features'] = 0
        return row

    centroid = np.nanmean(cohort_matrix, axis=0)

    for i, name in enumerate(feature_names):
        row[f'system_centroid_{name}'] = float(centroid[i])

    feature_stds = np.nanstd(cohort_matrix, axis=0)
    n_active = int(np.sum(np.isfinite(feature_stds) & (feature_stds > 1e-15)))
    row['n_active_features'] = n_active

    if n_cohorts > 1 and n_active > 0:
        diff = cohort_matrix - centroid[np.newaxis, :]
        diff = np.where(np.isfinite(diff), diff, 0.0)
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        row['system_dispersion_mean'] = float(np.mean(distances))
        row['system_dispersion_max'] = float(np.max(distances))
        row['system_dispersion_std'] = float(np.std(distances))
    else:
        row['system_dispersion_mean'] = 0.0
        row['system_dispersion_max'] = 0.0
        row['system_dispersion_std'] = 0.0

    return row
