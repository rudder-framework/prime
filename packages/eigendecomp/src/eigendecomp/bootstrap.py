"""
Jackknife confidence intervals for effective dimensionality.

Leave-one-out jackknife: remove each signal in turn, recompute effective_dim.
The spread of jackknife estimates gives a confidence interval.

This tells you whether the measured effective_dim is robust to removing
any single signal, or if one signal is dominating the geometry.
"""

import numpy as np
from typing import Dict


def jackknife_effective_dim(
    signal_matrix: np.ndarray,
    norm_method: str = "zscore",
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Jackknife confidence interval for effective dimensionality.

    Parameters
    ----------
    signal_matrix : np.ndarray
        (n_signals, n_features) feature matrix.
    norm_method : str
        Normalization method ("zscore" or "none").
    confidence_level : float
        Confidence level for interval (default 0.95).

    Returns
    -------
    dict with:
        effective_dim : float — full-sample estimate
        ci_lower : float — lower bound
        ci_upper : float — upper bound
        std : float — jackknife standard error
        n_jackknife : int — number of resamples
        min_estimate : float — minimum jackknife estimate
        max_estimate : float — maximum jackknife estimate
    """
    from eigendecomp.decompose import compute_eigendecomp

    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    N = signal_matrix.shape[0]

    if N < 4:  # need at least 3 after removing one
        return {
            'effective_dim': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_jackknife': 0,
            'min_estimate': np.nan,
            'max_estimate': np.nan,
        }

    # Full-sample estimate
    full = compute_eigendecomp(signal_matrix, norm_method=norm_method, min_signals=2)
    full_ed = full['effective_dim']

    # Leave-one-out estimates
    estimates = []
    for i in range(N):
        subset = np.delete(signal_matrix, i, axis=0)
        result = compute_eigendecomp(subset, norm_method=norm_method, min_signals=2)
        ed = result['effective_dim']
        if np.isfinite(ed):
            estimates.append(ed)

    if len(estimates) < 3:
        return {
            'effective_dim': full_ed,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_jackknife': len(estimates),
            'min_estimate': np.nan,
            'max_estimate': np.nan,
        }

    estimates = np.array(estimates)
    jack_mean = np.mean(estimates)
    jack_std = np.sqrt((N - 1) / N * np.sum((estimates - jack_mean) ** 2))

    # Normal approximation CI
    # z = 1.96 for 95% confidence
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence_level, 1.960)
    ci_lower = float(full_ed - z * jack_std)
    ci_upper = float(full_ed + z * jack_std)

    return {
        'effective_dim': full_ed,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': float(jack_std),
        'n_jackknife': len(estimates),
        'min_estimate': float(np.min(estimates)),
        'max_estimate': float(np.max(estimates)),
    }
