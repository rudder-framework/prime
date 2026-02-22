"""
State-space velocity field computation.

Takes a wide matrix (rows=time, cols=signals) and computes:
v(I) = x(I+1) - x(I)           velocity vector
s(I) = |v(I)|                   speed
d(I) = v(I) / |v(I)|            direction (unit vector)
a(I) = v(I+1) - v(I)            acceleration vector
κ(I) = |a_perp| / |v|²          curvature (how sharply turning)
motion_dim = exp(H(d²))         how many signals participate in motion
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_velocity_field(
    matrix: np.ndarray,
    signal_ids: List[str],
    indices: Optional[np.ndarray] = None,
    smooth_window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Compute state-space velocity field from wide observation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        (n_timesteps, n_signals) — wide format, rows=time, cols=signals.
    signal_ids : list of str
        Signal names (column labels).
    indices : np.ndarray, optional
        Index values (I) for each row. If None, uses 0..n-1.
    smooth_window : int
        Savitzky-Golay-style smoothing window (1 = no smoothing).

    Returns
    -------
    list of dict — one per valid timestep (n-2 rows due to differencing).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    N, D = matrix.shape

    if N < 3 or D < 1:
        return []

    if indices is None:
        indices = np.arange(N, dtype=float)

    # Handle NaN: fill with column mean
    for j in range(D):
        col = matrix[:, j]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            col[nans] = np.nanmean(col)
        elif nans.all():
            col[:] = 0.0
        matrix[:, j] = col

    # Optional smoothing
    if smooth_window > 2 and N > smooth_window:
        try:
            from scipy.signal import savgol_filter
            for j in range(D):
                matrix[:, j] = savgol_filter(matrix[:, j], smooth_window, min(2, smooth_window - 1))
        except ImportError:
            # Simple moving average fallback
            kernel = np.ones(smooth_window) / smooth_window
            for j in range(D):
                matrix[:, j] = np.convolve(matrix[:, j], kernel, mode='same')

    # Velocity: first difference
    v = np.diff(matrix, axis=0)  # (N-1, D)
    speed = np.linalg.norm(v, axis=1)  # (N-1,)

    # Direction: normalized velocity
    direction = np.zeros_like(v)
    nonzero = speed > 1e-12
    direction[nonzero] = v[nonzero] / speed[nonzero, np.newaxis]

    # Acceleration: second difference
    a = np.diff(v, axis=0)  # (N-2, D)
    accel_mag = np.linalg.norm(a, axis=1)

    results = []

    for i in range(len(a)):
        v_hat = direction[i]
        spd = speed[i]

        # Decompose acceleration: parallel + perpendicular to velocity
        a_parallel_scalar = float(np.dot(a[i], v_hat))
        a_perp = a[i] - a_parallel_scalar * v_hat
        a_perp_mag = float(np.linalg.norm(a_perp))

        # Curvature: |a_perp| / |v|²
        curvature = a_perp_mag / (spd ** 2 + 1e-12) if spd > 1e-12 else 0.0

        # Dominant motion signal
        dominant_idx = int(np.argmax(np.abs(v[i])))
        dominant_signal = signal_ids[dominant_idx] if dominant_idx < len(signal_ids) else ''
        dominant_fraction = float(np.abs(v[i, dominant_idx]) / (spd + 1e-12)) if spd > 1e-12 else 0.0

        # Motion dimensionality: exp(entropy of squared direction components)
        dir_sq = direction[i] ** 2 + 1e-12
        dir_sq = dir_sq / dir_sq.sum()
        motion_dim = float(np.exp(-np.sum(dir_sq * np.log(dir_sq + 1e-12))))

        results.append({
            'I': float(indices[i + 1]),
            'speed': float(spd),
            'acceleration_magnitude': float(accel_mag[i]),
            'acceleration_parallel': a_parallel_scalar,
            'acceleration_perpendicular': a_perp_mag,
            'curvature': curvature,
            'dominant_motion_signal': dominant_signal,
            'dominant_motion_fraction': dominant_fraction,
            'motion_dimensionality': motion_dim,
        })

    return results
