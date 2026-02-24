"""
PR8: Robust CONSTANT Detection Fix
===================================

Fixes false CONSTANT classification by using multiple criteria:
1. Absolute variance check (signal_std)
2. Relative variance check (coefficient of variation)
3. Unique value ratio check
4. Value range check

A signal is only CONSTANT if it TRULY has no information content.
If in doubt, it's NOT constant - let Manifold compute and find boring results.

Bug fixed: SKAB dataset signals with std=0.004-0.7 were incorrectly
classified as CONSTANT due to overly aggressive thresholds.
"""

from typing import Dict, Any, Optional
import math


# ============================================================
# CONSTANT Detection Configuration
# ============================================================

CONSTANT_CONFIG = {
    # Absolute variance: signal_std must be nearly zero
    # 1e-9 is essentially floating point noise
    'signal_std_max': 1e-9,
    
    # Relative variance: coefficient of variation (std/|mean|)
    # Even a tiny CV means there's SOME variation worth analyzing
    'cv_max': 1e-6,
    
    # Unique ratio: fraction of unique values
    # < 0.1% unique suggests categorical/constant
    # But ALSO requires low CV to confirm
    'unique_ratio_max': 0.001,
    
    # Value range: (max - min) / |mean|
    # If range is tiny relative to scale, it's constant
    'range_ratio_max': 1e-6,
    
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
    
    Uses multiple criteria to avoid false positives:
    1. Absolute std near zero
    2. Coefficient of variation near zero  
    3. Unique ratio very low AND confirmed by CV
    4. Value range near zero relative to scale
    
    Philosophy: When in doubt, return False. Let Manifold compute.
    A false negative (running engines on constant) wastes compute.
    A false positive (skipping engines on real signal) loses information.
    
    Args:
        signal_std: Standard deviation of signal
        signal_mean: Mean of signal
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
    
    # Check 1: Absolute standard deviation
    # If std is essentially zero, it's constant
    if signal_std < cfg['signal_std_max']:
        return True
    
    # Check 2: Coefficient of variation (scale-invariant)
    # Handles signals with large mean but tiny relative variation
    if signal_mean is not None and not math.isnan(signal_mean):
        mean_abs = abs(signal_mean)
        if mean_abs > 1e-10:  # Avoid division by zero
            cv = signal_std / mean_abs
            if cv < cfg['cv_max']:
                return True
    
    # Check 3: Unique ratio + CV confirmation
    # Low unique ratio alone is NOT sufficient (could be discrete/quantized)
    # Must ALSO have tiny CV to confirm constant
    if unique_ratio is not None and not math.isnan(unique_ratio):
        if unique_ratio < cfg['unique_ratio_max']:
            # Confirm with CV check
            if signal_mean is not None and not math.isnan(signal_mean):
                mean_abs = abs(signal_mean)
                if mean_abs > 1e-10:
                    cv = signal_std / mean_abs
                    # More lenient CV for unique_ratio trigger
                    if cv < 0.001:  # 0.1% relative variation
                        return True
    
    # Check 4: Value range (if provided)
    if value_range is not None and not math.isnan(value_range):
        if signal_mean is not None and not math.isnan(signal_mean):
            mean_abs = abs(signal_mean)
            if mean_abs > 1e-10:
                range_ratio = value_range / mean_abs
                if range_ratio < cfg['range_ratio_max']:
                    return True
    
    # Default: NOT constant
    # Let Manifold compute and produce boring-but-valid results
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
        unique_ratio=0.0001,  # 1 unique value
        n_samples=1000,
    )
    results.append((
        "true_constant",
        result == True,
        f"Expected True, got {result}"
    ))
    
    # Case 2: SKAB Accelerometer1RMS - NOT constant
    result = is_constant_signal(
        signal_std=0.00474,
        signal_mean=0.2126,
        unique_ratio=0.738,
        n_samples=9405,
    )
    results.append((
        "skab_accelerometer",
        result == False,
        f"Expected False, got {result} (std=0.00474, mean=0.21)"
    ))
    
    # Case 3: SKAB Temperature - NOT constant  
    result = is_constant_signal(
        signal_std=0.667,
        signal_mean=89.47,
        unique_ratio=0.812,
        n_samples=9405,
    )
    results.append((
        "skab_temperature",
        result == False,
        f"Expected False, got {result} (std=0.667, mean=89.5)"
    ))
    
    # Case 4: SKAB Thermocouple - NOT constant
    result = is_constant_signal(
        signal_std=0.731,
        signal_mean=28.47,
        unique_ratio=0.672,
        n_samples=9405,
    )
    results.append((
        "skab_thermocouple", 
        result == False,
        f"Expected False, got {result} (std=0.731, mean=28.5)"
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
        result == True,
        f"Expected True, got {result}"
    ))
    
    # Case 6: Large mean, tiny relative variation - constant
    result = is_constant_signal(
        signal_std=0.0001,
        signal_mean=1000000.0,  # 1 million
        unique_ratio=0.0001,
        n_samples=1000,
    )
    results.append((
        "large_scale_constant",
        result == True,
        f"Expected True, got {result} (CV = 1e-10)"
    ))
    
    # Case 7: Small signal with real variation - NOT constant
    result = is_constant_signal(
        signal_std=0.001,
        signal_mean=0.01,
        unique_ratio=0.5,
        n_samples=1000,
    )
    results.append((
        "small_varying",
        result == False,
        f"Expected False, got {result} (CV = 0.1 = 10%)"
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
        result == True,
        f"Expected True, got {result}"
    ))
    
    return results

