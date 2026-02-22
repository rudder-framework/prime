"""
Dimensional collapse detection.

Collapse = sustained negative velocity in effective_dim,
indicating the system is losing degrees of freedom.

This is the key failure signature: as a turbofan degrades,
signals become increasingly correlated (coupling increases),
effective_dim drops, and eigenvalues concentrate onto fewer modes.

Detection method:
1. Compute effective_dim velocity (from dynamics module)
2. Find runs of consecutive negative velocity below threshold
3. Collapse onset = first point of sustained decline
4. Collapse fraction = onset_index / total_length

IMPORTANT: This module COMPUTES indices and fractions.
It does NOT classify or interpret. Prime's SQL does that.
"""

import numpy as np
from typing import Dict, Any


def detect_collapse(
    effective_dim_velocity: np.ndarray,
    threshold_velocity: float = -0.05,
    min_run_length: int = 3,
    sustained_fraction: float = 0.3,
) -> Dict[str, Any]:
    """
    Detect dimensional collapse from effective_dim velocity series.

    Parameters
    ----------
    effective_dim_velocity : np.ndarray
        Velocity of effective_dim (from dynamics.compute_derivatives).
    threshold_velocity : float
        Velocity below this = collapsing. Default -0.05.
    min_run_length : int
        Minimum consecutive points below threshold for a collapse event.
    sustained_fraction : float
        Fraction of total points that must be below threshold
        for collapse to be flagged.

    Returns
    -------
    dict with:
        collapse_detected : bool — any sustained collapse found
        collapse_onset_idx : int or None — first index of sustained decline
        collapse_onset_fraction : float or None — onset_idx / total_length
        max_run_length : int — longest consecutive run below threshold
        fraction_below : float — fraction of points below threshold
        min_velocity : float — most negative velocity observed
        mean_velocity_below : float — mean velocity of below-threshold points
    """
    v = np.asarray(effective_dim_velocity, dtype=np.float64)
    n = len(v)

    if n < min_run_length:
        return _empty_collapse_result()

    # Find points below threshold (ignoring NaN)
    below = np.where(np.isfinite(v), v < threshold_velocity, False)
    n_valid = np.isfinite(v).sum()
    n_below = int(below.sum())

    if n_below == 0:
        return {
            'collapse_detected': False,
            'collapse_onset_idx': None,
            'collapse_onset_fraction': None,
            'max_run_length': 0,
            'fraction_below': 0.0,
            'min_velocity': float(np.nanmin(v)) if np.any(np.isfinite(v)) else np.nan,
            'mean_velocity_below': np.nan,
        }

    # Find runs of consecutive True
    runs = _find_runs(below)

    # Filter to runs >= min_run_length
    sustained_runs = [(start, length) for start, length in runs if length >= min_run_length]

    max_run = max(length for _, length in runs) if runs else 0

    # Collapse onset = start of first sustained run
    if sustained_runs:
        onset_idx = sustained_runs[0][0]
        onset_fraction = float(onset_idx) / n if n > 0 else None
    else:
        onset_idx = None
        onset_fraction = None

    fraction_below = n_below / n_valid if n_valid > 0 else 0.0
    collapse_detected = bool((len(sustained_runs) > 0) and (fraction_below >= sustained_fraction))

    # Stats on below-threshold points
    below_values = v[below & np.isfinite(v)]
    mean_below = float(np.mean(below_values)) if len(below_values) > 0 else np.nan

    return {
        'collapse_detected': collapse_detected,
        'collapse_onset_idx': onset_idx,
        'collapse_onset_fraction': onset_fraction,
        'max_run_length': max_run,
        'fraction_below': float(fraction_below),
        'min_velocity': float(np.nanmin(v)) if np.any(np.isfinite(v)) else np.nan,
        'mean_velocity_below': mean_below,
    }


def _find_runs(mask: np.ndarray):
    """Find consecutive runs of True in a boolean array.

    Returns list of (start_index, run_length) tuples.
    """
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


def _empty_collapse_result() -> Dict[str, Any]:
    return {
        'collapse_detected': False,
        'collapse_onset_idx': None,
        'collapse_onset_fraction': None,
        'max_run_length': 0,
        'fraction_below': 0.0,
        'min_velocity': np.nan,
        'mean_velocity_below': np.nan,
    }
