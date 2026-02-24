"""
FTLE computation per signal.

Delegates Lyapunov computation to pmtvs. This module handles:
- Windowing of raw signals for rolling FTLE
- Forward and backward FTLE
- Confidence estimation
- Embedding parameter caching (d2_onset-aware)

Math lives in pmtvs (lyapunov_rosenstein, ftle_local_linearization).
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple

_MAX_SAMPLES = 2000  # Cap for O(n²) Rosenstein / embedding param estimation


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


def _estimate_embedding_params(values: np.ndarray) -> Tuple[int, int]:
    """
    Estimate embedding parameters (tau, dim) from a signal.
    Caps input at _MAX_SAMPLES to keep O(n²) FNN tractable.
    """
    sample = _cap_tail(values)
    try:
        from pmtvs import optimal_delay, optimal_dimension
        tau = int(optimal_delay(sample))
        dim = int(optimal_dimension(sample))
    except (ImportError, Exception):
        tau = 1
        dim = 3
    dim = max(2, min(dim, 10))
    tau = max(1, min(tau, len(values) // 4))
    return tau, dim


def _embed(values: np.ndarray, tau: int, dim: int) -> Optional[np.ndarray]:
    """Time-delay embedding with pre-computed params."""
    try:
        from pmtvs import time_delay_embedding
        return time_delay_embedding(values, dim, tau)
    except (ImportError, Exception):
        n = len(values)
        if n < dim * tau:
            return None
        rows = n - (dim - 1) * tau
        embedded = np.zeros((rows, dim))
        for d in range(dim):
            embedded[:, d] = values[d * tau: d * tau + rows]
        return embedded


def _pmtvs_embedding(values, dim=None, tau=None):
    """Time-delay embedding via pmtvs. Computes params from capped subsample."""
    if tau is None or dim is None:
        est_tau, est_dim = _estimate_embedding_params(values)
        if tau is None:
            tau = est_tau
        if dim is None:
            dim = est_dim
    dim = max(2, min(dim, 10))
    tau = max(1, min(tau, len(values) // 4))
    embedded = _embed(values, tau, dim)
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

    Uses Rust (pmtvs_dynamics._core.ftle) when available.

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

    # Embedding
    embedded, dim, tau = _pmtvs_embedding(values)

    # Try Rust Jacobian-based FTLE
    try:
        from pmtvs_dynamics import HAS_RUST
        if HAS_RUST and embedded is not None and len(embedded) > 10:
            from pmtvs_dynamics import ftle as rust_ftle
            fwd_val, bwd_val = rust_ftle(embedded, dt=1, integration_time=1.0)
            ftle_val = float(fwd_val)
            confidence = 1.0 if np.isfinite(ftle_val) else 0.0
            return {
                'ftle': ftle_val,
                'confidence': confidence,
                'n_samples': len(values),
                'embedding_dim': dim,
                'embedding_tau': tau,
                'method': 'jacobian',
                'direction': direction,
            }
    except (ImportError, Exception):
        pass

    # Python fallback: Lyapunov exponent
    ftle_val = _pmtvs_lyapunov(values, method=method)
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


def _compute_ftle_rolling_rust(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Rust fast-path for rolling FTLE via pmtvs_dynamics._core.

    Returns None if Rust unavailable or embedding fails.
    """
    try:
        from pmtvs_dynamics import HAS_RUST
        if not HAS_RUST:
            return None
        from pmtvs_dynamics import rolling_ftle
    except (ImportError, AttributeError):
        return None

    # Embed the full signal once
    embedded, dim, tau = _pmtvs_embedding(values)
    if embedded is None or len(embedded) < window_size:
        return None

    # Rust rolling_ftle: Rayon-parallel Jacobian FTLE over all windows
    indices, fwd, bwd = rolling_ftle(
        embedded, window_size=window_size, stride=stride,
        dt=1, forward=True, backward=True,
    )

    indices = np.asarray(indices)
    fwd = np.asarray(fwd)
    bwd = np.asarray(bwd)

    results = []
    for i in range(len(indices)):
        start = int(indices[i])
        ftle_val = float(fwd[i])
        results.append({
            'ftle': ftle_val,
            'confidence': 1.0 if np.isfinite(ftle_val) else 0.0,
            'n_samples': window_size,
            'embedding_dim': dim,
            'embedding_tau': tau,
            'method': 'jacobian',
            'direction': 'forward',
            'I': start + window_size // 2,
            'window_start': start,
            'window_end': start + window_size,
        })
    return results


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

    Uses Rust (pmtvs_dynamics._core.rolling_ftle) when available for ~100x speedup.
    Falls back to pure Python Rosenstein method otherwise.

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

    # Try Rust fast-path first
    rust_result = _compute_ftle_rolling_rust(values, window_size, stride)
    if rust_result is not None:
        return rust_result

    # Python fallback
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
