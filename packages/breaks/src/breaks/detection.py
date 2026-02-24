"""
Structural break detection: CUSUM and Pettitt tests.

CUSUM: Detects mean shift by tracking cumulative deviations from the mean.
    S(t) = Σ(x(i) - μ) for i=1..t
    Break = index where |S(t)| is maximized.

Pettitt: Nonparametric change-point test.
    Tests H0: no change point vs H1: change at some τ.
    Based on Mann-Whitney-like rank statistic.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def detect_breaks_cusum(
    values: np.ndarray,
    threshold_sigma: float = 2.0,
    min_segment: int = 20,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CUSUM break detection.

    Parameters
    ----------
    values : np.ndarray
        1D time series.
    threshold_sigma : float
        Break significance threshold in standard deviations.
    min_segment : int
        Minimum segment length on each side of break.

    Returns
    -------
    dict with:
        break_detected : bool
        break_index : int or None
        cusum_max : float — maximum |S(t)|
        cusum_significance : float — cusum_max / (n * std)
        cusum_series : np.ndarray — cumulative sum series
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 2 * min_segment:
        return _empty_cusum_result()

    mu = np.mean(values)
    std = np.std(values)
    if std < 1e-30:
        return _empty_cusum_result()

    # Cumulative sum of deviations from mean
    cusum = np.cumsum(values - mu)

    # Find break point: maximum absolute deviation
    # Only consider valid positions (min_segment from each end)
    valid_range = cusum[min_segment: n - min_segment]
    if len(valid_range) == 0:
        return _empty_cusum_result()

    abs_cusum = np.abs(valid_range)
    break_local = int(np.argmax(abs_cusum))
    break_index = break_local + min_segment
    cusum_max = float(abs_cusum[break_local])

    # Significance: normalize by sqrt(n) * std
    significance = cusum_max / (np.sqrt(n) * std)
    break_detected = bool(significance > threshold_sigma)

    return {
        'break_detected': break_detected,
        'break_index': break_index if break_detected else None,
        'cusum_max': cusum_max,
        'cusum_significance': float(significance),
        'cusum_series': cusum,
    }


def detect_breaks_pettitt(
    values: np.ndarray,
    alpha: float = 0.05,
    min_segment: int = 20,
) -> Dict[str, Any]:
    """
    Pettitt nonparametric change-point test.

    Parameters
    ----------
    values : np.ndarray
        1D time series.
    alpha : float
        Significance level.
    min_segment : int
        Minimum segment length.

    Returns
    -------
    dict with:
        break_detected : bool
        break_index : int or None
        test_statistic : float — max |U(t)|
        p_value : float
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    values = values[np.isfinite(values)]
    n = len(values)

    if n < 2 * min_segment:
        return _empty_pettitt_result()

    # Compute U(t) = 2*R(t) - t*(n+1) where R(t) = sum of ranks of first t values
    # Equivalent: U(t) = Σ_i=1..t Σ_j=t+1..n sign(x(j) - x(i))
    U = np.zeros(n)
    for t in range(1, n):
        # Incremental: U(t) = U(t-1) + Σ_j sign(x(t) - x(j)) for all j≠t
        signs = np.sign(values[t] - values[:t]).sum() - np.sign(values[t] - values[t + 1:]).sum()
        U[t] = U[t - 1] + signs

    # Find break in valid range
    valid_U = np.abs(U[min_segment: n - min_segment])
    if len(valid_U) == 0:
        return _empty_pettitt_result()

    break_local = int(np.argmax(valid_U))
    break_index = break_local + min_segment
    K = float(np.max(valid_U))

    # Approximate p-value: p ≈ 2 * exp(-6K² / (n³ + n²))
    p_value = 2.0 * np.exp(-6.0 * K ** 2 / (n ** 3 + n ** 2))
    p_value = min(1.0, max(0.0, float(p_value)))

    break_detected = bool(p_value < alpha)

    return {
        'break_detected': break_detected,
        'break_index': break_index if break_detected else None,
        'test_statistic': K,
        'p_value': p_value,
    }


def _empty_cusum_result() -> Dict[str, Any]:
    return {
        'break_detected': False,
        'break_index': None,
        'cusum_max': 0.0,
        'cusum_significance': 0.0,
        'cusum_series': np.array([]),
    }


def _empty_pettitt_result() -> Dict[str, Any]:
    return {
        'break_detected': False,
        'break_index': None,
        'test_statistic': 0.0,
        'p_value': 1.0,
    }
