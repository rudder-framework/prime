"""
Distribution divergence measures between signals.

KL divergence: asymmetric information gap D_KL(P||Q)
JS divergence: symmetric, bounded [0, ln2] version
"""

import numpy as np


def kl_divergence(
    p_values: np.ndarray,
    q_values: np.ndarray,
    n_bins: int = 30,
) -> float:
    """KL divergence D_KL(P||Q) from histograms of two signals."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    p_values = p_values[np.isfinite(p_values)]
    q_values = q_values[np.isfinite(q_values)]

    if len(p_values) < 10 or len(q_values) < 10:
        return np.nan

    lo = min(p_values.min(), q_values.min())
    hi = max(p_values.max(), q_values.max())
    if hi - lo < 1e-12:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_values, bins=bins, density=True)
    q_hist, _ = np.histogram(q_values, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def js_divergence(
    p_values: np.ndarray,
    q_values: np.ndarray,
    n_bins: int = 30,
) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    p_values = p_values[np.isfinite(p_values)]
    q_values = q_values[np.isfinite(q_values)]

    if len(p_values) < 10 or len(q_values) < 10:
        return np.nan

    lo = min(p_values.min(), q_values.min())
    hi = max(p_values.max(), q_values.max())
    if hi - lo < 1e-12:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_values, bins=bins, density=True)
    q_hist, _ = np.histogram(q_values, bins=bins, density=True)

    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    m = 0.5 * (p_hist + q_hist)
    jsd = 0.5 * np.sum(p_hist * np.log(p_hist / m)) + 0.5 * np.sum(q_hist * np.log(q_hist / m))
    return float(jsd)
