"""
Flatten eigendecomp results to parquet-ready rows.

Eigendecomp returns arrays (eigenvalues, explained_ratio) and matrices
(principal_components, signal_loadings). This module flattens everything
into a single dict of scalars suitable for writing to parquet.
"""

import numpy as np
from typing import Dict, Any


def flatten_result(
    result: Dict[str, Any],
    max_eigenvalues: int = 5,
    include_loadings: bool = False,
) -> Dict[str, float]:
    """
    Flatten eigendecomp result dict to scalar key-value pairs.

    Parameters
    ----------
    result : dict
        Output from compute_eigendecomp().
    max_eigenvalues : int
        Number of eigenvalues/ratios to include.
    include_loadings : bool
        If True, include flattened PC loadings (large!).

    Returns
    -------
    dict of {str: float} suitable for a parquet row.
    """
    row = {}

    # Window index if present
    if 'I' in result:
        row['I'] = result['I']

    # Scalar metrics
    for key in ['effective_dim', 'effective_dim_entropy', 'total_variance',
                'condition_number', 'eigenvalue_entropy',
                'eigenvalue_entropy_normalized', 'ratio_2_1', 'ratio_3_1',
                'energy_concentration', 'n_signals', 'n_features',
                'n_features_valid']:
        val = result.get(key)
        if val is not None:
            row[key] = float(val) if not isinstance(val, (int, np.integer)) else int(val)

    # Eigenvalues
    eigenvalues = result.get('eigenvalues')
    if eigenvalues is not None:
        for i in range(min(max_eigenvalues, len(eigenvalues))):
            row[f'eigenvalue_{i}'] = float(eigenvalues[i])

    # Explained ratios
    explained = result.get('explained_ratio')
    if explained is not None:
        for i in range(min(max_eigenvalues, len(explained))):
            row[f'explained_ratio_{i}'] = float(explained[i])

    # Cumulative variance
    if explained is not None:
        cum = 0.0
        for i in range(min(max_eigenvalues, len(explained))):
            cum += float(explained[i]) if np.isfinite(explained[i]) else 0.0
            row[f'cumulative_variance_{i}'] = cum

    # PC loadings (optional, produces many columns)
    if include_loadings and result.get('principal_components') is not None:
        pcs = result['principal_components']
        n_pcs = min(3, pcs.shape[0])  # top 3 PCs only
        n_feat = pcs.shape[1]
        for pc_i in range(n_pcs):
            for feat_j in range(n_feat):
                row[f'pc{pc_i}_feat{feat_j}'] = float(pcs[pc_i, feat_j])

    return row


def flatten_batch(
    results: list,
    max_eigenvalues: int = 5,
    include_loadings: bool = False,
) -> list:
    """Flatten a list of eigendecomp results."""
    return [flatten_result(r, max_eigenvalues, include_loadings) for r in results]
