"""
Robust CONSTANT Detection
==========================

A signal is CONSTANT only if it truly has no dynamics:
  - 1-2 unique values (truly flat)
  - Zero standard deviation or zero range

NEVER uses coefficient of variation (CV = std/mean). CV is scale-dependent
and misclassifies signals with large means as CONSTANT even when they have
hundreds of unique values and real dynamics.

Philosophy: When in doubt, return False. Let Manifold compute.
A false positive (skipping engines on real signal) loses information forever.
"""

from typing import Dict, Any, Optional
import math


# ============================================================
# CONSTANT Detection Configuration
# ============================================================

CONSTANT_CONFIG = {
    # Absolute variance: signal_std must be nearly zero
    # 1e-10 is essentially floating point noise
    'signal_std_max': 1e-10,

    # Maximum unique values to be considered constant
    'n_unique_max': 2,

    # Minimum samples to make determination
    'min_samples': 10,
}


def is_constant_signal(
    signal_std: Optional[float],
    signal_mean: Optional[float],
    unique_ratio: Optional[float],
    n_samples: Optional[int],
    value_range: Optional[float] = None,
) -> bool:
    """
    Determine if a signal is truly CONSTANT (no information content).

    Uses n_unique + std — never CV. CV penalizes signals with large means.

    Args:
        signal_std: Standard deviation of signal
        signal_mean: Mean of signal (unused, kept for API compat)
        unique_ratio: Fraction of unique values (0-1)
        n_samples: Number of samples
        value_range: Optional (max - min) of signal

    Returns:
        True only if signal is definitively constant
    """
    cfg = CONSTANT_CONFIG

    # Not enough data to determine
    if n_samples is None or n_samples < cfg['min_samples']:
        return False

    # Handle missing/invalid values - when in doubt, not constant
    if signal_std is None or math.isnan(signal_std):
        return False

    # Primary check: n_unique_values
    if unique_ratio is not None and not math.isnan(unique_ratio):
        n_unique = max(1, round(unique_ratio * n_samples))
        if n_unique <= cfg['n_unique_max']:
            return True

    # Safety net: zero standard deviation
    if signal_std < cfg['signal_std_max']:
        return True

    # Safety net: zero range
    if value_range is not None and not math.isnan(value_range) and value_range == 0:
        return True

    # Everything else has dynamics — let Manifold compute
    return False


def classify_constant_from_row(row: Dict[str, Any]) -> bool:
    """
    Classify CONSTANT from a typology_raw row.
    
    Args:
        row: Dictionary with signal measures
        
    Returns:
        True if signal should be classified as CONSTANT
    """
    return is_constant_signal(
        signal_std=row.get('signal_std'),
        signal_mean=row.get('signal_mean'),
        unique_ratio=row.get('unique_ratio'),
        n_samples=row.get('n_samples'),
        value_range=row.get('value_range'),  # May not exist
    )


# ============================================================
# Validation against known cases
# ============================================================

def validate_constant_detection():
    """
    Validate CONSTANT detection against known cases.

    Returns list of (test_name, passed, message) tuples.
    """
    results = []

    # Case 1: True constant (all same value)
    result = is_constant_signal(
        signal_std=0.0,
        signal_mean=100.0,
        unique_ratio=0.0001,  # ~1 unique value
        n_samples=1000,
    )
    results.append((
        "true_constant",
        result is True,
        f"Expected True, got {result}"
    ))

    # Case 2: SKAB Accelerometer1RMS - NOT constant (738 unique values)
    result = is_constant_signal(
        signal_std=0.00474,
        signal_mean=0.2126,
        unique_ratio=0.738,
        n_samples=9405,
    )
    results.append((
        "skab_accelerometer",
        result is False,
        f"Expected False, got {result} (6941 unique values)"
    ))

    # Case 3: SKAB Temperature - NOT constant (7635 unique values)
    result = is_constant_signal(
        signal_std=0.667,
        signal_mean=89.47,
        unique_ratio=0.812,
        n_samples=9405,
    )
    results.append((
        "skab_temperature",
        result is False,
        f"Expected False, got {result} (7635 unique values)"
    ))

    # Case 4: SKAB Thermocouple - NOT constant (6320 unique values)
    result = is_constant_signal(
        signal_std=0.731,
        signal_mean=28.47,
        unique_ratio=0.672,
        n_samples=9405,
    )
    results.append((
        "skab_thermocouple",
        result is False,
        f"Expected False, got {result} (6320 unique values)"
    ))

    # Case 5: Near-zero signal that IS constant
    result = is_constant_signal(
        signal_std=1e-12,
        signal_mean=0.0,
        unique_ratio=0.001,
        n_samples=1000,
    )
    results.append((
        "zero_constant",
        result is True,
        f"Expected True, got {result}"
    ))

    # Case 6: Large mean, few unique values - constant
    # 0.0001 * 1000 = 0.1 → round → 0 → max(1,0) = 1 unique
    result = is_constant_signal(
        signal_std=0.0001,
        signal_mean=1000000.0,
        unique_ratio=0.0001,
        n_samples=1000,
    )
    results.append((
        "large_scale_constant",
        result is True,
        f"Expected True, got {result} (1 unique value)"
    ))

    # Case 7: Small signal with real variation - NOT constant (500 unique)
    result = is_constant_signal(
        signal_std=0.001,
        signal_mean=0.01,
        unique_ratio=0.5,
        n_samples=1000,
    )
    results.append((
        "small_varying",
        result is False,
        f"Expected False, got {result} (500 unique values)"
    ))

    # Case 8: Electrochemistry Mn_II (all zeros)
    result = is_constant_signal(
        signal_std=0.0,
        signal_mean=0.0,
        unique_ratio=0.0,
        n_samples=500,
    )
    results.append((
        "electrochemistry_mn_ii",
        result is True,
        f"Expected True, got {result}"
    ))

    # Case 9: C-MAPSS NRc — large mean, 195 unique values — NOT constant
    result = is_constant_signal(
        signal_std=5.0,
        signal_mean=8000.0,
        unique_ratio=195.0 / 192,  # >100% unique
        n_samples=192,
    )
    results.append((
        "cmapss_nrc",
        result is False,
        f"Expected False, got {result} (195 unique values, mean=8000)"
    ))

    # Case 10: C-MAPSS NRf — large mean, 27 unique values — NOT constant
    result = is_constant_signal(
        signal_std=0.05,
        signal_mean=2388.0,
        unique_ratio=27.0 / 192,
        n_samples=192,
    )
    results.append((
        "cmapss_nrf",
        result is False,
        f"Expected False, got {result} (27 unique values, mean=2388)"
    ))

    return results

