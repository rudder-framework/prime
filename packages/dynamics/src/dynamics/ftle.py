"""
FTLE computation per signal.

Delegates Lyapunov computation to pmtvs. This module handles:
- Windowing of raw signals for rolling FTLE
- Forward and backward FTLE
- Confidence estimation

Math lives in pmtvs (lyapunov_rosenstein, ftle_local_linearization).
"""

import numpy as np
from typing import Dict, Any, Optional, List

_MAX_SAMPLES = 2000  # Cap for O(n²) Rosenstein


def _cap_tail(values: np.ndarray, max_n: int = _MAX_SAMPLES) -> np.ndarray:
    """Take tail of signal if it exceeds max_n."""
    return values[-max_n:] if len(values) > max_n else values


def _pmtvs_lyapunov(values, method='rosenstein'):
    """Delegate to pmtvs Lyapunov computation."""
    try:
        if method == 'rosenstein':
            from pmtvs import lyapunov_rosenstein
            return float(lyapunov_rosenstein(values))
        else:
            from pmtvs import lyapunov_kantz
            return float(lyapunov_kantz(values))
    except (ImportError, Exception):
        return np.nan


def _pmtvs_ftle(trajectory, time_horizon=10):
    """Delegate to pmtvs FTLE."""
    try:
        from pmtvs import ftle_local_linearization
        ftle_vals, confidence = ftle_local_linearization(trajectory, time_horizon=time_horizon)
        return float(np.nanmean(ftle_vals)), float(np.nanmean(confidence))
    except (ImportError, Exception):
        return np.nan, np.nan


def _pmtvs_embedding(values, dim=None, tau=None):
    """Time-delay embedding via pmtvs."""
    try:
        from pmtvs import time_delay_embedding, optimal_delay, optimal_dimension
        if tau is None:
            tau = int(optimal_delay(values))
        if dim is None:
            dim = int(optimal_dimension(values))
        dim = max(2, min(dim, 10))
        tau = max(1, min(tau, len(values) // 4))
        return time_delay_embedding(values, dim, tau), dim, tau
    except (ImportError, Exception):
        # Minimal fallback embedding
        if tau is None:
            tau = 1
        if dim is None:
            dim = 3
        n = len(values)
        if n < dim * tau:
            return None, dim, tau
        rows = n - (dim - 1) * tau
        embedded = np.zeros((rows, dim))
        for d in range(dim):
            embedded[:, d] = values[d * tau: d * tau + rows]
        return embedded, dim, tau


def get_cache_strategy(d2_onset_pct: Optional[float] = None) -> dict:
    """
    Determine embedding parameter cache strategy from d2_onset_pct.

    Parameters
    ----------
    d2_onset_pct : float or None
        Fraction through signal where correlation dimension changes.
        From typology. None means unknown.

    Returns
    -------
    dict with:
        mode : str — 'lock', 'adaptive', or 'refresh_late'
        refresh_after_pct : float — fraction after which to start refreshing
        refresh_interval : int — windows between refreshes (0 = never)
    """
    if d2_onset_pct is None or d2_onset_pct > 0.95:
        return {'mode': 'lock', 'refresh_after_pct': 1.0, 'refresh_interval': 0}
    if d2_onset_pct >= 0.2:
        return {'mode': 'adaptive', 'refresh_after_pct': d2_onset_pct, 'refresh_interval': 5}
    return {'mode': 'refresh_late', 'refresh_after_pct': 0.0, 'refresh_interval': 10}


def compute_ftle(
    values: np.ndarray,
    method: str = 'rosenstein',
    direction: str = 'forward',
    min_samples: int = 200,
) -> Dict[str, Any]:
    """
    Compute FTLE for a single signal.

    Parameters
    ----------
    values : np.ndarray
        1D time series values.
    method : str
        'rosenstein' or 'kantz'.
    direction : str
        'forward' or 'backward'.
    min_samples : int
        Minimum samples required.

    Returns
    -------
    dict with:
        ftle : float — Lyapunov exponent estimate
        confidence : float — estimation confidence [0, 1]
        n_samples : int
        embedding_dim : int
        embedding_tau : int
        method : str
        direction : str
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    values = values[np.isfinite(values)]

    if len(values) < min_samples:
        return _empty_ftle_result(method, direction)

    values = _cap_tail(values)

    if direction == 'backward':
        values = values[::-1]

    # Lyapunov exponent
    ftle_val = _pmtvs_lyapunov(values, method=method)

    # Embedding for metadata
    embedded, dim, tau = _pmtvs_embedding(values)
    confidence = 1.0 if np.isfinite(ftle_val) else 0.0

    # FTLE from embedded trajectory if available
    if embedded is not None and len(embedded) > 50:
        ftle_embed, conf_embed = _pmtvs_ftle(embedded, time_horizon=min(10, len(embedded) // 5))
        if np.isfinite(ftle_embed):
            confidence = conf_embed

    return {
        'ftle': ftle_val,
        'confidence': confidence,
        'n_samples': len(values),
        'embedding_dim': dim,
        'embedding_tau': tau,
        'method': method,
        'direction': direction,
    }


def compute_ftle_rolling(
    values: np.ndarray,
    window_size: int = 500,
    stride: int = 50,
    method: str = 'rosenstein',
    min_samples: int = 200,
    d2_onset_pct: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Compute rolling FTLE over a signal with embedding parameter caching.

    Adjacent windows overlap ~94%, so embedding parameters (dim, tau) barely
    change. d2_onset_pct from typology tells us where signal structure shifts,
    so we lock params in the stable region and refresh only when needed.

    Parameters
    ----------
    values : np.ndarray
        Full 1D time series.
    window_size : int
        Rolling window size.
    stride : int
        Step between windows.
    min_samples : int
        Minimum samples per window.
    d2_onset_pct : float or None
        Fraction through signal where correlation dimension changes.
        None → lock mode (compute once, reuse forever).

    Returns
    -------
    list of dict — one per window, with 'I' for window center index.
    """
    values = np.asarray(values, dtype=np.float64).flatten()
    n = len(values)
    results = []

    strategy = get_cache_strategy(d2_onset_pct)
    mode = strategy['mode']
    refresh_after_pct = strategy['refresh_after_pct']
    refresh_interval = strategy['refresh_interval']

    cached_dim: Optional[int] = None
    cached_tau: Optional[int] = None
    windows_since_refresh = 0
    total_windows = max(1, (n - window_size) // stride + 1)

    for start in range(0, n - window_size + 1, stride):
        window = values[start:start + window_size]
        valid = window[np.isfinite(window)]

        if len(valid) < min_samples:
            continue

        valid = _cap_tail(valid)

        # Determine whether to refresh embedding params
        window_frac = start / max(1, n - window_size)
        should_refresh = False

        if cached_dim is None:
            # First window — always compute
            should_refresh = True
        elif mode == 'lock':
            should_refresh = False
        elif mode == 'adaptive':
            if window_frac >= refresh_after_pct:
                windows_since_refresh += 1
                if windows_since_refresh >= refresh_interval:
                    should_refresh = True
        elif mode == 'refresh_late':
            windows_since_refresh += 1
            if windows_since_refresh >= refresh_interval:
                should_refresh = True

        # Embedding
        if should_refresh:
            embedded, dim, tau = _pmtvs_embedding(valid)
            cached_dim = dim
            cached_tau = tau
            windows_since_refresh = 0
        else:
            embedded, dim, tau = _pmtvs_embedding(valid, dim=cached_dim, tau=cached_tau)

        # Lyapunov exponent (always computed — it's the actual measurement)
        ftle_val = _pmtvs_lyapunov(valid, method=method)
        confidence = 1.0 if np.isfinite(ftle_val) else 0.0

        # FTLE from embedded trajectory if available
        if embedded is not None and len(embedded) > 50:
            ftle_embed, conf_embed = _pmtvs_ftle(embedded, time_horizon=min(10, len(embedded) // 5))
            if np.isfinite(ftle_embed):
                confidence = conf_embed

        result = {
            'ftle': ftle_val,
            'confidence': confidence,
            'n_samples': len(valid),
            'embedding_dim': dim,
            'embedding_tau': tau,
            'method': method,
            'direction': 'forward',
            'I': start + window_size // 2,
            'window_start': start,
            'window_end': start + window_size,
        }
        results.append(result)

    return results


def _empty_ftle_result(method: str, direction: str) -> Dict[str, Any]:
    return {
        'ftle': np.nan,
        'confidence': 0.0,
        'n_samples': 0,
        'embedding_dim': 0,
        'embedding_tau': 0,
        'method': method,
        'direction': direction,
    }
