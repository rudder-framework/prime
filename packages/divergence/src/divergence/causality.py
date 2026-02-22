"""
Granger causality and transfer entropy between signal pairs.

Granger: Does knowing A's past improve prediction of B beyond B's own past?
Transfer entropy: How much information flows from A to B?

Both are directional: compute(A, B) ≠ compute(B, A).
"""

import numpy as np
from typing import Dict, Any

_MAX_SAMPLES = 2000


def compute_granger(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
    min_samples: int = 50,
) -> Dict[str, float]:
    """
    Compute Granger causality: does x Granger-cause y?

    Parameters
    ----------
    x, y : np.ndarray
        1D time series (same length, aligned in time).
    max_lag : int
        Maximum lag for VAR model.
    min_samples : int
        Minimum samples required.

    Returns
    -------
    dict with granger_f, granger_p, best_lag.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    # Remove NaN (pairwise)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < min_samples or len(x) < max_lag + 10:
        return {'granger_f': np.nan, 'granger_p': np.nan, 'best_lag': 0}

    # Cap for tractability
    if len(x) > _MAX_SAMPLES:
        x, y = x[-_MAX_SAMPLES:], y[-_MAX_SAMPLES:]

    try:
        from pmtvs.pairwise.causality import granger_causality
        result = granger_causality(x, y, max_lag=max_lag)
        return {
            'granger_f': float(result.get('f_statistic', np.nan)),
            'granger_p': float(result.get('p_value', np.nan)),
            'best_lag': int(result.get('best_lag', 0)),
        }
    except (ImportError, Exception):
        pass

    # Minimal VAR-based Granger test fallback
    best_f, best_p, best_lag = np.nan, np.nan, 0
    n = len(x)

    for lag in range(1, max_lag + 1):
        if n - lag < lag + 10:
            continue

        # Restricted model: y(t) = a0 + a1*y(t-1) + ... + aL*y(t-L)
        Y_target = y[lag:]
        Y_restricted = np.column_stack([y[lag - k: n - k] for k in range(1, lag + 1)])
        Y_restricted = np.column_stack([np.ones(len(Y_target)), Y_restricted])

        # Unrestricted model: add x lags
        X_lags = np.column_stack([x[lag - k: n - k] for k in range(1, lag + 1)])
        Y_unrestricted = np.column_stack([Y_restricted, X_lags])

        try:
            # OLS for both models
            _, rss_r, _, _ = np.linalg.lstsq(Y_restricted, Y_target, rcond=None)
            _, rss_u, _, _ = np.linalg.lstsq(Y_unrestricted, Y_target, rcond=None)

            rss_r = float(rss_r[0]) if len(rss_r) > 0 else float(np.sum((Y_target - Y_restricted @ np.linalg.lstsq(Y_restricted, Y_target, rcond=None)[0]) ** 2))
            rss_u = float(rss_u[0]) if len(rss_u) > 0 else float(np.sum((Y_target - Y_unrestricted @ np.linalg.lstsq(Y_unrestricted, Y_target, rcond=None)[0]) ** 2))

            df1 = lag
            df2 = len(Y_target) - Y_unrestricted.shape[1]

            if df2 > 0 and rss_u > 1e-30:
                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                if np.isfinite(f_stat) and f_stat > 0:
                    if np.isnan(best_f) or f_stat > best_f:
                        best_f = float(f_stat)
                        best_lag = lag
                        # Approximate p-value
                        try:
                            from scipy.stats import f as f_dist
                            best_p = float(1 - f_dist.cdf(f_stat, df1, df2))
                        except ImportError:
                            best_p = np.nan
        except (np.linalg.LinAlgError, ValueError):
            continue

    return {'granger_f': best_f, 'granger_p': best_p, 'best_lag': best_lag}


def compute_transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
) -> Dict[str, float]:
    """
    Compute transfer entropy from x to y.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Parameters
    ----------
    x, y : np.ndarray
        1D time series.
    lag : int
        Time lag.
    n_bins : int
        Histogram bins for discretization.

    Returns
    -------
    dict with transfer_entropy.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < lag + 20:
        return {'transfer_entropy': np.nan}

    try:
        from pmtvs.pairwise.causality import transfer_entropy
        te = transfer_entropy(x, y, lag=lag, n_bins=n_bins)
        return {'transfer_entropy': float(te)}
    except (ImportError, Exception):
        pass

    # Histogram-based fallback
    y_future = y[lag:]
    y_past = y[:-lag]
    x_past = x[:-lag]

    def _entropy_hist(*arrays, bins=n_bins):
        if len(arrays) == 1:
            hist, _ = np.histogram(arrays[0], bins=bins)
        else:
            sample = np.column_stack(arrays)
            hist, _ = np.histogramdd(sample, bins=bins)
        hist = hist.flatten()
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p + 1e-30))

    h_yf_yp = _entropy_hist(y_future, y_past) - _entropy_hist(y_past)
    h_yf_yp_xp = _entropy_hist(y_future, y_past, x_past) - _entropy_hist(y_past, x_past)
    te = max(0, h_yf_yp - h_yf_yp_xp)

    return {'transfer_entropy': float(te)}
