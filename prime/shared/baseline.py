"""
Adaptive baseline discovery for 1D time series.

Slides a window across the signal, scores each position by stability (low variance),
and returns the most stable region -- wherever it falls. No assumption about
which fraction of data is "healthy."

For multi-column geometry baseline discovery, see prime/cohorts/baseline.py.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StableBaseline:
    """Result of adaptive baseline discovery."""
    start_idx: int
    end_idx: int
    baseline_mean: float
    baseline_std: float
    stability_score: float      # higher = more stable (1/std)
    fraction_start: float       # start as fraction of total (0.0 = beginning)
    fraction_end: float
    is_early: bool              # True if baseline is in first 30%
    values: np.ndarray          # the actual baseline slice


def find_stable_baseline(
    values: np.ndarray,
    window_fraction: float = 0.2,
    min_window: int = 10,
    stride_fraction: float = 0.05,
) -> StableBaseline:
    """
    Find the most stable region in a 1D time series.

    Slides a window and scores each position by:
        stability_score = 1.0 / (std + epsilon)

    The window with the highest stability score is the baseline.

    Args:
        values: 1D numpy array (velocity, energy, any scalar series)
        window_fraction: baseline window size as fraction of total length
        min_window: minimum absolute window size
        stride_fraction: how far to slide each step (fraction of total)

    Returns:
        StableBaseline with discovered region
    """
    n = len(values)

    # Short signal: entire thing is the baseline
    if n < min_window:
        return StableBaseline(
            start_idx=0,
            end_idx=n,
            baseline_mean=float(np.nanmean(values)),
            baseline_std=float(np.nanstd(values)),
            stability_score=1.0 / (float(np.nanstd(values)) + 1e-12),
            fraction_start=0.0,
            fraction_end=1.0,
            is_early=True,
            values=values,
        )

    window_size = max(min_window, int(n * window_fraction))
    stride = max(1, int(n * stride_fraction))

    best_score = -1.0
    best_start = 0

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = values[start:end]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        std = float(np.std(valid))
        score = 1.0 / (std + 1e-12)
        if score > best_score:
            best_score = score
            best_start = start

    best_end = best_start + window_size
    baseline_slice = values[best_start:best_end]

    return StableBaseline(
        start_idx=best_start,
        end_idx=best_end,
        baseline_mean=float(np.nanmean(baseline_slice)),
        baseline_std=float(np.nanstd(baseline_slice)),
        stability_score=best_score,
        fraction_start=best_start / n,
        fraction_end=best_end / n,
        is_early=(best_start / n) < 0.3,
        values=baseline_slice,
    )
