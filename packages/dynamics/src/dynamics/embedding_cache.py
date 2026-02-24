"""
FTLE Embedding Cache
====================
Caches embedding parameters (delay τ, dimension m) to avoid recomputing
them for every rolling window. Uses d2_onset_pct from typology to decide
cache strategy.

The Problem:
    Rolling FTLE currently computes embedding parameters (delay τ from 
    ACF/mutual information, dimension m from false nearest neighbors) 
    for every rolling window. FNN is O(n²) per window. For Rössler 
    with 470 windows × 4 signals = 1,880 embedding computations.
    Most produce the same answer because signal structure isn't changing.

The Solution:
    Cache embedding parameters. Recompute only when signal structure 
    changes. Use d2_onset_pct from typology to decide when.

Three Cache Modes:
    GLOBAL  (d2_onset = null/NaN): Signal is stationary (chaotic, periodic, 
            noise). Compute embedding ONCE, reuse for entire signal.
            Example: Rössler attractor, healthy turbofan, building vibrations.

    SPLIT   (d2_onset >= 0.4): Late onset. Signal is stable for a long time, 
            then changes. Cache for stable region, refresh once after onset,
            cache again for remainder.
            Example: Turbofan with d2_onset at 82% of life.

    PERIODIC (d2_onset < 0.4): Early onset. Signal changes throughout most 
            of its life. Refresh embedding every N windows.
            Example: Rapidly degrading system, early fault.

Usage:
    from ftle_embedding_cache import EmbeddingCache

    cache = EmbeddingCache(
        d2_onset_pct=manifest['cohorts']['system']['x']['typology']['d2_onset_pct'],
        refresh_interval=20  # for PERIODIC mode
    )

    for window_idx, window_data in enumerate(rolling_windows):
        tau, m = cache.get_or_compute(window_idx, n_windows, window_data)
        embedded = delay_embed(window_data, tau, m)
        ftle_value = compute_ftle(embedded)
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple, Callable


class CacheMode(Enum):
    GLOBAL = "global"       # Compute once, never refresh
    SPLIT = "split"         # Cache per region (stable / active)
    PERIODIC = "periodic"   # Refresh every N windows


class EmbeddingCache:
    """
    Caches embedding parameters (tau, dim) for rolling FTLE computation.
    
    Selects cache strategy based on d2_onset_pct from typology.
    
    Parameters
    ----------
    d2_onset_pct : float or None/NaN
        Where D2 onset occurs in signal life (0.0 to 1.0).
        None or NaN = stationary signal, use GLOBAL mode.
    onset_threshold : float
        d2_onset_pct below this triggers PERIODIC mode. Default: 0.4
    refresh_interval : int
        Windows between refreshes in PERIODIC mode. Default: 20
    compute_fn : callable, optional
        Function that computes (tau, dim) from a 1D signal array.
        Signature: compute_fn(values: np.ndarray) -> Tuple[int, int]
        If None, uses default AMI + FNN estimation.
    """
    
    def __init__(
        self,
        d2_onset_pct: Optional[float] = None,
        onset_threshold: float = 0.4,
        refresh_interval: int = 20,
        compute_fn: Optional[Callable] = None,
    ):
        self.d2_onset_pct = d2_onset_pct
        self.onset_threshold = onset_threshold
        self.refresh_interval = refresh_interval
        self.compute_fn = compute_fn or _default_embedding_params
        
        # Determine mode
        if d2_onset_pct is None or (isinstance(d2_onset_pct, float) and np.isnan(d2_onset_pct)):
            self.mode = CacheMode.GLOBAL
        elif d2_onset_pct >= onset_threshold:
            self.mode = CacheMode.SPLIT
        else:
            self.mode = CacheMode.PERIODIC
        
        # Cache storage
        self._cached_tau: Optional[int] = None
        self._cached_dim: Optional[int] = None
        self._cache_region: Optional[str] = None  # 'stable' or 'active' for SPLIT
        self._last_refresh_window: int = -999
        self._compute_count: int = 0
    
    @property
    def compute_count(self) -> int:
        """How many times embedding was actually computed (vs cached)."""
        return self._compute_count
    
    def get_or_compute(
        self,
        window_idx: int,
        n_windows: int,
        window_data: np.ndarray,
    ) -> Tuple[int, int]:
        """
        Get cached embedding parameters or compute fresh ones.
        
        Parameters
        ----------
        window_idx : int
            Current window index (0-based)
        n_windows : int
            Total number of rolling windows
        window_data : np.ndarray
            Raw signal values for this window
            
        Returns
        -------
        (tau, dim) : Tuple[int, int]
            Delay and embedding dimension
        """
        if self.mode == CacheMode.GLOBAL:
            return self._get_global(window_data)
        elif self.mode == CacheMode.SPLIT:
            return self._get_split(window_idx, n_windows, window_data)
        else:
            return self._get_periodic(window_idx, window_data)
    
    def _get_global(self, window_data: np.ndarray) -> Tuple[int, int]:
        """GLOBAL: Compute once, cache forever."""
        if self._cached_tau is None:
            self._cached_tau, self._cached_dim = self._compute(window_data)
        return self._cached_tau, self._cached_dim
    
    def _get_split(
        self,
        window_idx: int,
        n_windows: int,
        window_data: np.ndarray,
    ) -> Tuple[int, int]:
        """SPLIT: One cache for stable region, one for active region."""
        # Determine which region this window is in
        life_pct = window_idx / max(n_windows - 1, 1)
        current_region = 'stable' if life_pct < self.d2_onset_pct else 'active'
        
        if self._cache_region != current_region:
            # Region changed — recompute
            self._cached_tau, self._cached_dim = self._compute(window_data)
            self._cache_region = current_region
        
        return self._cached_tau, self._cached_dim
    
    def _get_periodic(
        self,
        window_idx: int,
        window_data: np.ndarray,
    ) -> Tuple[int, int]:
        """PERIODIC: Refresh every N windows."""
        if (window_idx - self._last_refresh_window) >= self.refresh_interval:
            self._cached_tau, self._cached_dim = self._compute(window_data)
            self._last_refresh_window = window_idx
        
        return self._cached_tau, self._cached_dim
    
    def _compute(self, window_data: np.ndarray) -> Tuple[int, int]:
        """Actually compute embedding parameters. Expensive."""
        self._compute_count += 1
        return self.compute_fn(window_data)
    
    def summary(self) -> dict:
        """Return cache statistics."""
        return {
            'mode': self.mode.value,
            'd2_onset_pct': self.d2_onset_pct,
            'compute_count': self._compute_count,
            'cached_tau': self._cached_tau,
            'cached_dim': self._cached_dim,
        }


def _default_embedding_params(values: np.ndarray) -> Tuple[int, int]:
    """
    Compute embedding parameters using AMI for tau and FNN for dim.
    Falls back to simpler methods if pmtvs functions unavailable.
    
    Parameters
    ----------
    values : np.ndarray
        1D signal array
        
    Returns
    -------
    (tau, dim) : Tuple[int, int]
    """
    tau = _estimate_tau(values)
    dim = _estimate_dim(values, tau)
    return tau, dim


def _estimate_tau(values: np.ndarray) -> int:
    """
    Estimate delay tau via first minimum of automutual information.
    Falls back to first zero-crossing of ACF.
    """
    try:
        from pmtvs.information import auto_mutual_information
        ami = auto_mutual_information(values, max_lag=min(len(values) // 4, 100))
        # First minimum
        for i in range(1, len(ami) - 1):
            if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
                return max(i, 1)
        return max(len(ami) // 4, 1)
    except (ImportError, Exception):
        pass
    
    # Fallback: first zero-crossing of ACF
    n = len(values)
    mean = np.mean(values)
    var = np.var(values)
    if var == 0:
        return 1
    
    centered = values - mean
    max_lag = min(n // 4, 100)
    
    for lag in range(1, max_lag):
        acf = np.mean(centered[:n - lag] * centered[lag:]) / var
        if acf <= 0:
            return max(lag, 1)
    
    return max(max_lag // 4, 1)


def _estimate_dim(values: np.ndarray, tau: int) -> int:
    """
    Estimate embedding dimension via false nearest neighbors.
    Falls back to Cao's method if FNN unavailable.
    """
    try:
        from pmtvs.embedding import false_nearest_neighbors
        fnn_fracs = false_nearest_neighbors(values, tau=tau, max_dim=10)
        # First dimension where FNN < 5%
        for dim, frac in enumerate(fnn_fracs, start=1):
            if frac < 0.05:
                return dim
        return len(fnn_fracs)
    except (ImportError, Exception):
        pass
    
    # Fallback: Cao's method
    n = len(values)
    max_dim = min(10, n // (2 * tau) - 1)
    if max_dim < 2:
        return 2
    
    prev_e1 = None
    for dim in range(1, max_dim):
        # Build delay vectors
        m = n - dim * tau
        if m < 10:
            return dim
        
        vectors = np.zeros((m, dim))
        for d in range(dim):
            vectors[:, d] = values[d * tau:d * tau + m]
        
        # Mean distance to nearest neighbor
        # Subsample for speed
        sample_size = min(200, m)
        indices = np.random.choice(m, sample_size, replace=False)
        
        distances = []
        for idx in indices:
            diffs = np.abs(vectors - vectors[idx])
            chebyshev = np.max(diffs, axis=1)
            chebyshev[idx] = np.inf  # exclude self
            nn_dist = np.min(chebyshev)
            if nn_dist > 0:
                distances.append(nn_dist)
        
        e1 = np.mean(distances) if distances else 1.0
        
        if prev_e1 is not None:
            ratio = e1 / prev_e1 if prev_e1 > 0 else 1.0
            if abs(ratio - 1.0) < 0.1:  # Saturated
                return dim
        
        prev_e1 = e1
    
    return max_dim


# ============================================================
# INTEGRATION HELPER: Wire into orchestration pipeline
# ============================================================

def create_cache_for_signal(
    signal_id: str,
    manifest: dict,
    cohort: str = 'system',
    refresh_interval: int = 20,
) -> EmbeddingCache:
    """
    Create an EmbeddingCache for a signal using its manifest config.
    
    Parameters
    ----------
    signal_id : str
        Signal name (e.g., 'x', 'NRf')
    manifest : dict
        Parsed manifest.yaml
    cohort : str
        Cohort name. Default: 'system'
    refresh_interval : int
        Windows between refreshes in PERIODIC mode
        
    Returns
    -------
    EmbeddingCache
    """
    # Extract d2_onset_pct from manifest
    d2_onset = None
    try:
        sig_config = manifest['cohorts'][cohort][signal_id]
        typology = sig_config.get('typology', {})
        d2_onset = typology.get('d2_onset_pct')
        # Handle YAML .nan
        if d2_onset is not None and isinstance(d2_onset, float) and np.isnan(d2_onset):
            d2_onset = None
    except (KeyError, TypeError):
        pass
    
    return EmbeddingCache(
        d2_onset_pct=d2_onset,
        refresh_interval=refresh_interval,
    )


def rolling_ftle_with_cache(
    values: np.ndarray,
    rolling_window: int,
    rolling_stride: int,
    cache: EmbeddingCache,
    flow_time: Optional[int] = None,
    n_neighbors: Optional[int] = None,
) -> np.ndarray:
    """
    Compute rolling FTLE using cached embedding parameters.
    
    If Rust pmtvs_dynamics is available, embeds once and delegates 
    to Rust rolling_ftle. Otherwise falls back to Python with caching.
    
    Parameters
    ----------
    values : np.ndarray
        Raw 1D signal (full length, not windowed)
    rolling_window : int
        Points per FTLE window
    rolling_stride : int
        Step between windows
    cache : EmbeddingCache
        Pre-configured cache for this signal
    flow_time : int, optional
        Steps forward for evolution. Default: rolling_window // 4
    n_neighbors : int, optional
        Neighbors for Jacobian. Default: 2 * dim + 1
        
    Returns
    -------
    np.ndarray
        FTLE values, one per rolling window
    """
    n = len(values)
    n_windows = max(1, (n - rolling_window) // rolling_stride + 1)
    
    if flow_time is None:
        flow_time = rolling_window // 4
    
    # ==========================================
    # RUST FAST PATH: embed once, parallel FTLE
    # ==========================================
    try:
        from pmtvs_dynamics._core import rolling_ftle as rust_rolling_ftle
        
        # Get embedding params from first window (or cached)
        first_window = values[:min(rolling_window, n)]
        tau, dim = cache.get_or_compute(0, n_windows, first_window)
        
        if n_neighbors is None:
            n_neighbors = 2 * dim + 1
        
        # Embed FULL signal once
        embedded = _delay_embed(values, tau, dim)
        
        if embedded.shape[0] < rolling_window:
            return np.array([np.nan])
        
        # Delegate to Rust — it handles windowing and parallelism
        result = rust_rolling_ftle(
            embedded,
            rolling_window,
            rolling_stride,
            flow_time,
            n_neighbors,
        )
        return np.asarray(result)
        
    except ImportError:
        pass
    
    # ==========================================
    # PYTHON FALLBACK: cached embedding per window
    # ==========================================
    ftle_values = []
    
    for w in range(n_windows):
        start = w * rolling_stride
        end = start + rolling_window
        
        if end > n:
            break
        
        window_data = values[start:end]
        
        # Get cached or fresh embedding params
        tau, dim = cache.get_or_compute(w, n_windows, window_data)
        
        if n_neighbors is None:
            n_neighbors = 2 * dim + 1
        
        # Embed this window
        embedded = _delay_embed(window_data, tau, dim)
        
        # Compute FTLE for this window
        try:
            from pmtvs.dynamics import ftle_local_linearization
            ftle_val = ftle_local_linearization(embedded, flow_time=flow_time)
        except (ImportError, Exception):
            ftle_val = _simple_ftle(embedded, flow_time)
        
        ftle_values.append(ftle_val)
    
    return np.array(ftle_values)


def _delay_embed(values: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """Create delay-embedded matrix from 1D signal."""
    n = len(values)
    m = n - (dim - 1) * tau
    if m <= 0:
        return np.empty((0, dim))
    
    embedded = np.zeros((m, dim))
    for d in range(dim):
        embedded[:, d] = values[d * tau:d * tau + m]
    
    return embedded


def _simple_ftle(embedded: np.ndarray, flow_time: int) -> float:
    """Minimal FTLE computation for fallback."""
    n = embedded.shape[0]
    if n < flow_time + 2:
        return np.nan
    
    # Track divergence of nearby points
    mid = n // 2
    ref = embedded[mid]
    
    # Find nearest neighbor (excluding temporal neighbors)
    dists = np.linalg.norm(embedded - ref, axis=1)
    dists[max(0, mid - 10):min(n, mid + 10)] = np.inf
    nn_idx = np.argmin(dists)
    
    d0 = dists[nn_idx]
    if d0 < 1e-12:
        return np.nan
    
    # Evolved distance
    evolved_mid = min(mid + flow_time, n - 1)
    evolved_nn = min(nn_idx + flow_time, n - 1)
    df = np.linalg.norm(embedded[evolved_mid] - embedded[evolved_nn])
    
    if df < 1e-12:
        return 0.0
    
    return np.log(df / d0) / flow_time
