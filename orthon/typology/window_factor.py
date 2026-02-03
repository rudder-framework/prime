"""
Compute window_factor for each signal based on typology characteristics.

window_factor is a multiplier that scales engine base_window:
    effective_window = engine.base_window × signal.window_factor

Higher factor = signal needs larger windows for reliable analysis.
"""

import polars as pl
from typing import Dict, Any


def compute_window_factor(row: Dict[str, Any]) -> float:
    """
    Compute window_factor based on signal characteristics.

    Args:
        row: Dict with typology columns (temporal_pattern, spectral, hurst, etc.)

    Returns:
        window_factor (1.0 = baseline, >1.0 = needs larger window)
    """
    factor = 1.0

    # =========================================================================
    # SPECTRAL COMPLEXITY
    # =========================================================================
    spectral = row.get('spectral', 'BROADBAND')

    if spectral == 'NARROWBAND':
        # Need more samples to resolve spectral peaks
        factor *= 1.5
    elif spectral == 'ONE_OVER_F':
        # Low-frequency content needs longer observation
        factor *= 1.25
    elif spectral == 'HARMONIC':
        # Multiple peaks need good frequency resolution
        factor *= 1.4

    # =========================================================================
    # TEMPORAL PATTERN
    # =========================================================================
    temporal = row.get('temporal_pattern', 'RANDOM')

    if temporal == 'TRENDING':
        # Non-stationary needs context to see the trend
        factor *= 1.25
    elif temporal == 'QUASI_PERIODIC':
        # Need to capture the period
        factor *= 1.3
    elif temporal == 'PERIODIC':
        # Need multiple complete cycles
        factor *= 1.4
    elif temporal == 'CHAOTIC':
        # Chaotic systems need more samples for reliable statistics
        factor *= 1.5

    # =========================================================================
    # MEMORY / PERSISTENCE
    # =========================================================================
    hurst = row.get('hurst')

    if hurst is not None:
        if hurst < 0.4:
            # Anti-persistent (rough/noisy) - needs more averaging
            factor *= 1.3
        elif hurst > 0.8:
            # Highly persistent (smooth trends) - can use smaller windows
            factor *= 0.85

    # =========================================================================
    # COMPLEXITY / ENTROPY
    # =========================================================================
    perm_entropy = row.get('perm_entropy') or row.get('permutation_entropy')

    if perm_entropy is not None:
        if perm_entropy > 0.9:
            # High entropy = noisy = need more samples
            factor *= 1.2
        elif perm_entropy < 0.3:
            # Low entropy = predictable = smaller windows ok
            factor *= 0.9

    # =========================================================================
    # STATIONARITY
    # =========================================================================
    stationarity = row.get('stationarity', 'STATIONARY')

    if stationarity in ('NON_STATIONARY', 'DIFFERENCE_STATIONARY'):
        # Non-stationary needs larger context windows
        factor *= 1.2

    # =========================================================================
    # CLAMP TO REASONABLE RANGE
    # =========================================================================
    factor = max(0.5, min(3.0, factor))

    return round(factor, 2)


def add_window_factor(typology_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add window_factor column to typology DataFrame.

    Args:
        typology_df: Typology DataFrame with classification columns

    Returns:
        DataFrame with window_factor column added
    """
    # Convert to list of dicts, compute factor, convert back
    rows = typology_df.to_dicts()

    for row in rows:
        row['window_factor'] = compute_window_factor(row)

    return pl.DataFrame(rows)
