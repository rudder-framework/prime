"""
Signal-level pairwise computation.

For each window, takes the signal_vector matrix (n_signals × n_features)
and computes C(N,2) pairs. For 14 sensors → 91 pairs per window.

This captures the internal structure of the signal cloud:
which signals move together, which diverge, which are redundant.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Any


def compute_signal_pairwise(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    centroid: Optional[np.ndarray] = None,
    window_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise metrics for all signal pairs at one window.

    Parameters
    ----------
    signal_matrix : np.ndarray
        (n_signals, n_features) — one row per signal.
    signal_ids : list of str
        Signal identifiers, length n_signals.
    centroid : np.ndarray, optional
        Cohort centroid for context metrics.
    window_index : int, optional
        Window index (I) to include in output rows.

    Returns
    -------
    list of dict — one per pair, C(N,2) total.
    """
    from pairwise.metrics import compute_pair_metrics, compute_pair_metrics_with_context

    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    N = signal_matrix.shape[0]

    if N < 2:
        return []

    results = []
    indices = list(range(N))

    for i, j in combinations(indices, 2):
        if centroid is not None:
            row = compute_pair_metrics_with_context(
                signal_matrix[i], signal_matrix[j],
                centroid,
                signal_a=signal_ids[i], signal_b=signal_ids[j],
            )
        else:
            row = compute_pair_metrics(
                signal_matrix[i], signal_matrix[j],
                signal_a=signal_ids[i], signal_b=signal_ids[j],
            )

        if window_index is not None:
            row['I'] = window_index

        results.append(row)

    return results


def compute_signal_pairwise_batch(
    matrices: List[np.ndarray],
    signal_ids: List[str],
    centroids: Optional[List[np.ndarray]] = None,
    window_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise metrics across multiple windows.

    Parameters
    ----------
    matrices : list of np.ndarray
        Each (n_signals, n_features), one per window.
    signal_ids : list of str
        Signal identifiers (same for all windows).
    centroids : list of np.ndarray, optional
        Centroid per window.
    window_indices : list of int, optional
        Window index per entry.

    Returns
    -------
    list of dict — all pairs across all windows.
    """
    all_results = []

    for idx, matrix in enumerate(matrices):
        centroid = centroids[idx] if centroids is not None else None
        win_idx = window_indices[idx] if window_indices is not None else idx

        rows = compute_signal_pairwise(
            matrix, signal_ids,
            centroid=centroid,
            window_index=win_idx,
        )
        all_results.extend(rows)

    return all_results
