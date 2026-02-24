"""
Cohort Vector
=============
Aggregates signal-level features into cohort-level vectors.

The pivot: signals as rows → signals as columns.
At each window, N signals × D features becomes a matrix.
Centroid = mean position in feature space.
Dispersion = how spread out the signals are.

Usage:
    from vector.cohort import compute_cohort

    rows = compute_cohort(
        signal_vectors=signal_df,  # polars DataFrame
        cohort_id='unit_001',
        feature_columns=['statistics_kurtosis', 'hurst_exponent', ...],
    )
"""

import numpy as np
from typing import Dict, Any, List


def compute_cohort(
    signal_matrix: np.ndarray,
    cohort_id: str,
    window_index: int,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Compute cohort vector from signal matrix at one window.

    Args:
        signal_matrix: (n_signals, n_features) array.
            Rows = signals, columns = features.
            NaN allowed — will be handled with nanmean/nanstd.
        cohort_id: Cohort identifier.
        window_index: Which window this is.
        feature_names: Names of the feature columns.

    Returns:
        Dict with:
            cohort_id, window_index,
            centroid_{feature} for each feature,
            dispersion_mean, dispersion_max, dispersion_std,
            n_signals, n_active_features
    """
    n_signals, n_features = signal_matrix.shape

    row = {
        'cohort_id': cohort_id,
        'window_index': window_index,
        'n_signals': n_signals,
    }

    if n_signals == 0:
        for name in feature_names:
            row[f'centroid_{name}'] = float('nan')
        row['dispersion_mean'] = float('nan')
        row['dispersion_max'] = float('nan')
        row['dispersion_std'] = float('nan')
        row['n_active_features'] = 0
        return row

    # Centroid: mean across signals per feature
    centroid = np.nanmean(signal_matrix, axis=0)

    for i, name in enumerate(feature_names):
        row[f'centroid_{name}'] = float(centroid[i])

    # Count features with actual variance (not all-NaN or constant)
    feature_stds = np.nanstd(signal_matrix, axis=0)
    n_active = int(np.sum(np.isfinite(feature_stds) & (feature_stds > 1e-15)))
    row['n_active_features'] = n_active

    # Dispersion: distances from centroid
    if n_signals > 1 and n_active > 0:
        # Euclidean distance of each signal from centroid (NaN-safe)
        diff = signal_matrix - centroid[np.newaxis, :]
        # Replace NaN diffs with 0 (don't penalize missing features)
        diff = np.where(np.isfinite(diff), diff, 0.0)
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        row['dispersion_mean'] = float(np.mean(distances))
        row['dispersion_max'] = float(np.max(distances))
        row['dispersion_std'] = float(np.std(distances))
    else:
        row['dispersion_mean'] = 0.0
        row['dispersion_max'] = 0.0
        row['dispersion_std'] = 0.0

    return row


def pivot_to_matrix(
    signal_rows: List[Dict[str, Any]],
    feature_columns: List[str],
) -> np.ndarray:
    """
    Pivot signal vector rows at one window into an (n_signals, n_features) matrix.

    Args:
        signal_rows: List of dicts from signal.compute_signal(),
            all sharing the same window_index.
        feature_columns: Which keys to extract as features.

    Returns:
        (n_signals, n_features) numpy array.
    """
    n = len(signal_rows)
    d = len(feature_columns)
    matrix = np.full((n, d), np.nan)

    for i, row in enumerate(signal_rows):
        for j, col in enumerate(feature_columns):
            val = row.get(col)
            if val is not None:
                try:
                    matrix[i, j] = float(val)
                except (ValueError, TypeError):
                    pass

    return matrix
