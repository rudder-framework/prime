"""
Ridge proximity: urgency metric combining FTLE + velocity.

urgency = v · ∇FTLE
time_to_ridge = (ridge_threshold - ftle_current) / ftle_gradient

Computes per signal per timestep. Does NOT classify —
Prime SQL handles urgency_class (CRITICAL/ELEVATED/WARNING/NOMINAL).
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_ridge_proximity(
    ftle_series: np.ndarray,
    speed_series: np.ndarray,
    indices: Optional[np.ndarray] = None,
    ridge_threshold: float = 0.05,
    smooth_window: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute ridge proximity metrics from FTLE and speed series.

    Parameters
    ----------
    ftle_series : np.ndarray
        Rolling FTLE values over time (one per window).
    speed_series : np.ndarray
        Speed from velocity field (one per window).
    indices : np.ndarray, optional
        Index values. If None, uses 0..n-1.
    ridge_threshold : float
        FTLE value considered "near ridge".
    smooth_window : int
        Smoothing for gradient computation.

    Returns
    -------
    list of dict — one per valid timestep.
    """
    ftle = np.asarray(ftle_series, dtype=np.float64)
    speed = np.asarray(speed_series, dtype=np.float64)
    n = min(len(ftle), len(speed))
    ftle, speed = ftle[:n], speed[:n]

    if indices is None:
        indices = np.arange(n, dtype=float)

    if n < 3:
        return []

    # Optional smoothing
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        ftle_smooth = np.convolve(ftle, kernel, mode='same')
    else:
        ftle_smooth = ftle

    # FTLE gradient (central differences)
    ftle_grad = np.full(n, np.nan)
    ftle_grad[1:-1] = (ftle_smooth[2:] - ftle_smooth[:-2]) / 2.0
    ftle_grad[0] = ftle_smooth[1] - ftle_smooth[0]
    ftle_grad[-1] = ftle_smooth[-1] - ftle_smooth[-2]

    # FTLE acceleration
    ftle_accel = np.full(n, np.nan)
    if n >= 3:
        ftle_accel[1:-1] = ftle_smooth[2:] - 2 * ftle_smooth[1:-1] + ftle_smooth[:-2]

    results = []
    for i in range(n):
        ftle_cur = float(ftle[i])
        grad = float(ftle_grad[i])
        spd = float(speed[i])

        # Urgency: speed × ftle_gradient (positive = approaching ridge)
        urgency = spd * grad if np.isfinite(spd) and np.isfinite(grad) else np.nan

        # Time to ridge estimate
        if np.isfinite(grad) and grad > 1e-8 and ftle_cur < ridge_threshold:
            time_to_ridge = (ridge_threshold - ftle_cur) / grad
            time_to_ridge = max(0, float(time_to_ridge))
        else:
            time_to_ridge = np.nan

        results.append({
            'I': float(indices[i]),
            'ftle_current': ftle_cur,
            'ftle_gradient': grad,
            'ftle_acceleration': float(ftle_accel[i]),
            'speed': spd,
            'urgency': urgency if np.isfinite(urgency) else np.nan,
            'time_to_ridge': time_to_ridge,
        })

    return results
