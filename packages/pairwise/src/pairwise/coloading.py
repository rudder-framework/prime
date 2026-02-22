"""
Eigenvector co-loading analysis.

When two signals both load heavily on the same principal component,
they share variance in the same dimension. This flags pairs that
need expensive directional analysis (Granger causality) vs pairs
where simple correlation suffices.

High co-loading + high correlation → potential causal relationship → run Granger.
Low co-loading → independent signals → correlation is sufficient.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations


def compute_coloading_flags(
    signal_loadings: np.ndarray,
    signal_ids: List[str],
    threshold: float = 0.3,
    n_pcs: int = 3,
) -> List[Dict]:
    """
    Flag signal pairs with high eigenvector co-loading.

    Parameters
    ----------
    signal_loadings : np.ndarray
        (n_signals, n_components) — signal projections onto PCs.
        From eigendecomp result['signal_loadings'].
    signal_ids : list of str
        Signal identifiers.
    threshold : float
        Co-loading threshold. Pairs where both signals have
        |loading| > threshold on the same PC are flagged.
    n_pcs : int
        Number of PCs to check (default: top 3).

    Returns
    -------
    list of dict with keys:
        signal_a, signal_b : str
        needs_granger : bool — True if high co-loading detected
        max_coloading : float — maximum co-loading product across PCs
        coloading_pc : int — which PC has the highest co-loading
    """
    signal_loadings = np.asarray(signal_loadings, dtype=np.float64)
    N, K = signal_loadings.shape
    n_check = min(n_pcs, K)

    # Normalize loadings to unit variance per PC for comparability
    # (loadings from eigendecomp are projections, magnitudes vary)
    loading_std = np.std(signal_loadings[:, :n_check], axis=0)
    loading_std[loading_std < 1e-15] = 1.0
    normed = signal_loadings[:, :n_check] / loading_std

    results = []
    for i, j in combinations(range(N), 2):
        # Co-loading product: |loading_i * loading_j| per PC
        coloading = np.abs(normed[i] * normed[j])
        max_cl = float(np.max(coloading))
        max_pc = int(np.argmax(coloading))

        # Flag if both signals have substantial loading on same PC
        both_loaded = (np.abs(normed[i]) > threshold) & (np.abs(normed[j]) > threshold)
        needs_granger = bool(np.any(both_loaded))

        results.append({
            'signal_a': signal_ids[i],
            'signal_b': signal_ids[j],
            'needs_granger': needs_granger,
            'max_coloading': max_cl,
            'coloading_pc': max_pc,
        })

    return results


def merge_coloading_with_pairwise(
    pairwise_rows: List[Dict],
    coloading_rows: List[Dict],
) -> List[Dict]:
    """
    Merge co-loading flags into pairwise result rows.

    Matches on (signal_a, signal_b) pair. Adds needs_granger,
    max_coloading, coloading_pc to each pairwise row.
    """
    # Build lookup: (a, b) → coloading dict
    lookup = {}
    for row in coloading_rows:
        key = (row['signal_a'], row['signal_b'])
        lookup[key] = row
        # Also store reverse
        lookup[(row['signal_b'], row['signal_a'])] = row

    for row in pairwise_rows:
        key = (row.get('signal_a', ''), row.get('signal_b', ''))
        cl = lookup.get(key, {})
        row['needs_granger'] = cl.get('needs_granger', False)
        row['max_coloading'] = cl.get('max_coloading', 0.0)
        row['coloading_pc'] = cl.get('coloading_pc', -1)

    return pairwise_rows
