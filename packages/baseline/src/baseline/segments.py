"""
Segment comparison: early-life vs late-life geometry delta.

Splits a cohort's eigendecomp trajectory into early and late segments,
computes summary statistics for each, and reports the delta.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_segment_comparison(
    effective_dim: np.ndarray,
    eigenvalues: np.ndarray,
    total_variance: np.ndarray,
    split_fraction: float = 0.5,
) -> Dict[str, Any]:
    """
    Compare early vs late segments of a cohort.

    Parameters
    ----------
    effective_dim : np.ndarray
        Effective dimension per window.
    eigenvalues : np.ndarray
        Leading eigenvalue per window (eigenvalue_0).
    total_variance : np.ndarray
        Total variance per window.
    split_fraction : float
        Where to split (0.5 = median).

    Returns
    -------
    dict with early/late means and deltas for each metric.
    """
    eff = np.asarray(effective_dim, dtype=np.float64)
    eig = np.asarray(eigenvalues, dtype=np.float64)
    var = np.asarray(total_variance, dtype=np.float64)
    n = len(eff)

    if n < 4:
        return _empty_comparison()

    split = int(n * split_fraction)
    split = max(2, min(split, n - 2))

    result = {}
    for name, series in [('effective_dim', eff), ('eigenvalue_0', eig), ('total_variance', var)]:
        early = series[:split]
        late = series[split:]

        early_valid = early[np.isfinite(early)]
        late_valid = late[np.isfinite(late)]

        if len(early_valid) > 0 and len(late_valid) > 0:
            early_mean = float(np.mean(early_valid))
            late_mean = float(np.mean(late_valid))
            result[f'{name}_early'] = early_mean
            result[f'{name}_late'] = late_mean
            result[f'{name}_delta'] = late_mean - early_mean
            result[f'{name}_delta_pct'] = ((late_mean - early_mean) / (abs(early_mean) + 1e-30)) * 100
        else:
            result[f'{name}_early'] = np.nan
            result[f'{name}_late'] = np.nan
            result[f'{name}_delta'] = np.nan
            result[f'{name}_delta_pct'] = np.nan

    result['n_windows'] = n
    result['split_index'] = split

    return result


def _empty_comparison() -> Dict[str, Any]:
    result = {}
    for name in ['effective_dim', 'eigenvalue_0', 'total_variance']:
        result[f'{name}_early'] = np.nan
        result[f'{name}_late'] = np.nan
        result[f'{name}_delta'] = np.nan
        result[f'{name}_delta_pct'] = np.nan
    result['n_windows'] = 0
    result['split_index'] = 0
    return result
