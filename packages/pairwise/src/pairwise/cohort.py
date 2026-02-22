"""
Cohort-level pairwise computation.

Computes relationships between cohorts (e.g., different turbofan units).
Same metrics as signal-level but applied to cohort centroid vectors.

N cohorts → C(N,2) pairs per window.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Any


def compute_cohort_pairwise(
    cohort_vectors: np.ndarray,
    cohort_ids: List[str],
    window_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise metrics between all cohorts at one window.

    Parameters
    ----------
    cohort_vectors : np.ndarray
        (n_cohorts, n_features) — one row per cohort centroid.
    cohort_ids : list of str
        Cohort identifiers.
    window_index : int, optional
        Window index (I).

    Returns
    -------
    list of dict — one per pair.
    """
    from pairwise.metrics import compute_pair_metrics

    cohort_vectors = np.asarray(cohort_vectors, dtype=np.float64)
    N = cohort_vectors.shape[0]

    if N < 2:
        return []

    results = []
    for i, j in combinations(range(N), 2):
        row = compute_pair_metrics(
            cohort_vectors[i], cohort_vectors[j],
            signal_a=cohort_ids[i], signal_b=cohort_ids[j],
        )
        # Rename signal_a/b to cohort_a/b for clarity
        row['cohort_a'] = row.pop('signal_a')
        row['cohort_b'] = row.pop('signal_b')

        if window_index is not None:
            row['I'] = window_index

        results.append(row)

    return results
