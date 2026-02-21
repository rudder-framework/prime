"""
Transition Detection — Canary System for Character Shifts
==========================================================

Reads typology_vector.parquet (per-window metrics from Manifold Stage 00a)
and detects when a signal's character shifts significantly from its own
rolling baseline.

This is the "canary detector" — typology transitions precede engine metric
thresholds. A signal whose spectral_flatness spikes or whose hurst drops
is changing character BEFORE the downstream engines (FTLE, velocity, etc.)
would flag it.

Detection method:
  1. Per signal, compute rolling baseline (median over N trailing windows)
  2. Z-score each window against its rolling baseline
  3. Flag windows where |z| > threshold for >= min_consecutive windows
  4. Output: transition events with signal_id, window_id, metric, direction, magnitude

Prime classifies. This is classification — detecting WHEN character changes.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


# Default parameters
DEFAULT_ROLLING_WINDOW = 10    # trailing windows for baseline
DEFAULT_Z_THRESHOLD = 2.5      # z-score threshold for deviation
DEFAULT_MIN_CONSECUTIVE = 3    # minimum consecutive deviating windows

# Metrics to monitor for transitions
TRANSITION_METRICS = [
    'hurst', 'perm_entropy', 'sample_entropy', 'lyapunov_proxy',
    'spectral_flatness', 'kurtosis', 'cv', 'trend_strength',
]


def detect_transitions(
    typology_vector_path: str,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    verbose: bool = True,
) -> pl.DataFrame:
    """Detect character transitions in typology_vector data.

    Args:
        typology_vector_path: Path to typology_vector.parquet.
        rolling_window: Number of trailing windows for baseline computation.
        z_threshold: Z-score threshold for flagging deviation.
        min_consecutive: Minimum consecutive deviating windows to count as transition.
        verbose: Print progress.

    Returns:
        DataFrame of transition events with columns:
        signal_id, cohort, window_id, signal_0_center, metric,
        direction ('increase'|'decrease'), magnitude (z-score),
        baseline_value, current_value.
    """
    tv = pl.read_parquet(typology_vector_path)

    if verbose:
        print("=" * 70)
        print("TRANSITION DETECTION — Canary System")
        print(f"Rolling baseline: {rolling_window} windows, z-threshold: {z_threshold}")
        print(f"Min consecutive: {min_consecutive}")
        print("=" * 70)

    has_cohort = 'cohort' in tv.columns
    group_cols = ['signal_id']
    if has_cohort:
        group_cols.append('cohort')

    # Available metrics (intersect with what's in the data)
    metrics = [m for m in TRANSITION_METRICS if m in tv.columns]
    if verbose:
        print(f"Monitoring {len(metrics)} metrics: {metrics}")

    events = []

    # Process each signal (and cohort if present)
    groups = tv.group_by(group_cols)
    for group_keys, group_df in groups:
        if isinstance(group_keys, tuple):
            signal_id = group_keys[0]
            cohort = group_keys[1] if has_cohort else ''
        else:
            signal_id = group_keys
            cohort = ''

        # Sort by window_id
        sorted_df = group_df.sort('window_id')
        n_windows = sorted_df.height

        if n_windows < rolling_window + min_consecutive:
            continue

        for metric in metrics:
            values = sorted_df[metric].to_numpy().astype(float)

            # Skip if all NaN
            if np.all(np.isnan(values)):
                continue

            # Compute rolling baseline (median) and rolling std
            for i in range(rolling_window, n_windows):
                baseline_slice = values[max(0, i - rolling_window):i]
                valid_baseline = baseline_slice[~np.isnan(baseline_slice)]

                if len(valid_baseline) < 3:
                    continue

                baseline_median = np.median(valid_baseline)
                baseline_std = np.std(valid_baseline)

                if baseline_std < 1e-10:
                    continue

                current = values[i]
                if np.isnan(current):
                    continue

                z = (current - baseline_median) / baseline_std

                if abs(z) >= z_threshold:
                    # Check consecutive requirement
                    consecutive = 1
                    for j in range(i + 1, min(i + min_consecutive, n_windows)):
                        future_val = values[j]
                        if np.isnan(future_val):
                            break
                        future_z = (future_val - baseline_median) / baseline_std
                        if abs(future_z) >= z_threshold and np.sign(future_z) == np.sign(z):
                            consecutive += 1
                        else:
                            break

                    if consecutive >= min_consecutive:
                        window_row = sorted_df.row(i, named=True)
                        events.append({
                            'signal_id': signal_id,
                            'cohort': cohort,
                            'window_id': int(window_row['window_id']),
                            'signal_0_center': float(window_row['signal_0_center']),
                            'metric': metric,
                            'direction': 'increase' if z > 0 else 'decrease',
                            'magnitude': float(abs(z)),
                            'baseline_value': float(baseline_median),
                            'current_value': float(current),
                        })

    # Build output DataFrame
    schema = {
        'signal_id': pl.Utf8,
        'cohort': pl.Utf8,
        'window_id': pl.UInt32,
        'signal_0_center': pl.Float64,
        'metric': pl.Utf8,
        'direction': pl.Utf8,
        'magnitude': pl.Float64,
        'baseline_value': pl.Float64,
        'current_value': pl.Float64,
    }

    if events:
        result = pl.DataFrame(events, schema=schema)
    else:
        result = pl.DataFrame(schema=schema)

    if verbose:
        n_signals = result['signal_id'].n_unique() if result.height > 0 else 0
        print(f"\n  Transitions detected: {result.height} events across {n_signals} signals")
        if result.height > 0:
            metric_counts = result.group_by('metric').agg(pl.len().alias('count')).sort('count', descending=True)
            for row in metric_counts.iter_rows(named=True):
                print(f"    {row['metric']}: {row['count']} events")

    return result


def run(
    typology_vector_path: str,
    output_path: str = "transitions.parquet",
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run transition detection and write output.

    Args:
        typology_vector_path: Path to typology_vector.parquet.
        output_path: Where to write transitions.parquet.
        rolling_window: Trailing windows for baseline.
        z_threshold: Z-score deviation threshold.
        min_consecutive: Minimum consecutive windows.
        verbose: Print progress.

    Returns:
        Transitions DataFrame.
    """
    result = detect_transitions(
        typology_vector_path,
        rolling_window=rolling_window,
        z_threshold=z_threshold,
        min_consecutive=min_consecutive,
        verbose=verbose,
    )

    result.write_parquet(output_path)
    if verbose:
        print(f"  Wrote {output_path}")

    return result
