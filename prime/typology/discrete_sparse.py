"""
Level 2 Typology - Discrete & Sparse Classification
====================================================
PR #5: Adds detection for non-continuous signal types

These classifiers run BEFORE continuous classification (PR4).
Decision order:
    1. CONSTANT  (zero variance)
    2. BINARY    (exactly 2 values)
    3. DISCRETE  (few integer levels)
    4. IMPULSIVE (extreme kurtosis + crest factor)
    5. EVENT     (sparse + high kurtosis)
    6. STEP      (sparse derivative)
    7. INTERMITTENT (bursty activity)
    8. → Fall through to continuous classification (PR4)
"""

from typing import Any, Dict, Optional

from prime.config.discrete_sparse_config import (
    DISCRETE_SPARSE_CONFIG,
    DISCRETE_SPARSE_SPECTRAL,
    DISCRETE_SPARSE_ENGINES,
)
from prime.typology.constant_detection import classify_constant_from_row


def is_constant(row: Dict[str, Any]) -> bool:
    """
    Detect CONSTANT signal: zero or negligible variance.

    PR8 FIX: Uses robust multi-criteria detection:
    1. Absolute std near zero (< 1e-9)
    2. Coefficient of variation near zero (< 1e-6)
    3. Low unique ratio CONFIRMED by low CV

    Philosophy: When in doubt, return False. Let Manifold compute.
    A false positive (skipping real signal) loses information.
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('constant', {})
    if not cfg.get('enabled', True):
        return False

    # Use PR8 robust detection
    return classify_constant_from_row(row)


def is_binary(row: Dict[str, Any]) -> bool:
    """
    Detect BINARY signal: exactly two distinct values.

    Examples: on/off, 0/1, True/False
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('binary', {})
    if not cfg.get('enabled', True):
        return False

    unique_count = row.get('unique_count', None)
    if unique_count is None:
        # Try to infer from unique_ratio and n_samples
        unique_ratio = row.get('unique_ratio', 1.0)
        n_samples = row.get('n_samples', 1000)
        unique_count = int(unique_ratio * n_samples)

    target = cfg.get('unique_count_exact', 2)
    return unique_count == target


def is_discrete(row: Dict[str, Any]) -> bool:
    """
    Detect DISCRETE signal: integer values with few distinct levels.

    Examples: gear position (1-6), error codes, state IDs
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('discrete', {})
    if not cfg.get('enabled', True):
        return False

    is_integer = row.get('is_integer', False)
    unique_ratio = row.get('unique_ratio', 1.0)
    n_samples = row.get('n_samples', 1000)

    # Must be integer-valued
    if cfg.get('is_integer_required', True) and not is_integer:
        return False

    # Check unique ratio
    ur_max = cfg.get('unique_ratio_max', 0.05)
    if unique_ratio <= ur_max:
        return True

    # Or check absolute unique count
    unique_count = row.get('unique_count', int(unique_ratio * n_samples))
    uc_max = cfg.get('unique_count_max', 50)
    if unique_count <= uc_max:
        return True

    return False


def is_impulsive(row: Dict[str, Any]) -> bool:
    """
    Detect IMPULSIVE signal: rare extreme spikes dominate.

    Requires BOTH high kurtosis AND high crest factor.
    Examples: bearing impacts, flash crashes
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('impulsive', {})
    if not cfg.get('enabled', True):
        return False

    kurtosis = row.get('kurtosis', 0.0)
    crest_factor = row.get('crest_factor', 1.0)

    kurt_min = cfg.get('kurtosis_min', 20.0)
    crest_min = cfg.get('crest_factor_min', 10.0)

    return kurtosis >= kurt_min and crest_factor >= crest_min


def is_event(row: Dict[str, Any]) -> bool:
    """
    Detect EVENT signal: sparse with rare occurrences.

    Requires high sparsity AND moderate-high kurtosis.
    Examples: earthquake catalog, intrusion alerts
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('event', {})
    if not cfg.get('enabled', True):
        return False

    sparsity = row.get('sparsity', 0.0)
    kurtosis = row.get('kurtosis', 0.0)

    sparsity_min = cfg.get('sparsity_min', 0.80)
    kurt_min = cfg.get('kurtosis_min', 10.0)

    return sparsity >= sparsity_min and kurtosis >= kurt_min


def is_step(row: Dict[str, Any]) -> bool:
    """
    Detect STEP signal: piecewise constant with regime changes.

    The derivative (diff) is mostly zero.
    Examples: thermostat setpoint, control mode
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('step', {})
    if not cfg.get('enabled', True):
        return False

    # Need derivative sparsity (% of diff() that is zero)
    deriv_sparsity = row.get('derivative_sparsity', None)
    if deriv_sparsity is None:
        return False  # Can't determine without this measure

    unique_ratio = row.get('unique_ratio', 1.0)

    deriv_min = cfg.get('derivative_sparsity_min', 0.90)
    ur_max = cfg.get('unique_ratio_max', 0.10)

    return deriv_sparsity >= deriv_min and unique_ratio <= ur_max


def is_intermittent(row: Dict[str, Any]) -> bool:
    """
    Detect INTERMITTENT signal: alternating active/quiet periods.

    Has significant zero-runs but also significant activity.
    Examples: voice activity, rainfall, bursty network traffic
    """
    cfg = DISCRETE_SPARSE_CONFIG.get('intermittent', {})
    if not cfg.get('enabled', True):
        return False

    zero_run_ratio = row.get('zero_run_ratio', 0.0)
    sparsity = row.get('sparsity', 0.0)

    zrr_min = cfg.get('zero_run_ratio_min', 0.30)
    sparsity_range = cfg.get('sparsity_range', [0.30, 0.80])

    # Must have significant zero runs
    if zero_run_ratio < zrr_min:
        return False

    # Sparsity in middle range (not EVENT, not continuous)
    if sparsity < sparsity_range[0] or sparsity > sparsity_range[1]:
        return False

    return True


def classify_discrete_sparse(row: Dict[str, Any]) -> Optional[str]:
    """
    Classify discrete/sparse signal type.

    Returns:
        Type name (CONSTANT, BINARY, DISCRETE, etc.) or None if continuous

    Decision order matters - more specific types first:
        1. CONSTANT (zero variance - most restrictive)
        2. BINARY (exactly 2 values)
        3. DISCRETE (few integer levels)
        4. IMPULSIVE (extreme spikes)
        5. EVENT (sparse occurrences)
        6. STEP (piecewise constant)
        7. INTERMITTENT (bursty)
        8. None → continuous classification
    """
    if is_constant(row):
        return 'CONSTANT'

    if is_binary(row):
        return 'BINARY'

    if is_discrete(row):
        return 'DISCRETE'

    if is_impulsive(row):
        return 'IMPULSIVE'

    if is_event(row):
        return 'EVENT'

    if is_step(row):
        return 'STEP'

    if is_intermittent(row):
        return 'INTERMITTENT'

    return None  # Fall through to continuous classification


def get_spectral_for_discrete(temporal_pattern: str) -> str:
    """Get spectral classification for discrete/sparse type."""
    return DISCRETE_SPARSE_SPECTRAL.get(temporal_pattern.lower(), 'UNKNOWN')


def get_engines_for_discrete(temporal_pattern: str) -> Dict[str, list]:
    """Get engine adjustments for discrete/sparse type."""
    return DISCRETE_SPARSE_ENGINES.get(
        temporal_pattern.lower(),
        {'add': [], 'remove': []}
    )


def apply_discrete_sparse_classification(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply discrete/sparse classification to a row.

    If signal is discrete/sparse, sets temporal_pattern and spectral.
    If continuous, returns row unchanged (for PR4 to handle).

    Returns:
        Updated row dict with 'is_discrete_sparse' flag
    """
    row = dict(row)  # Copy

    discrete_type = classify_discrete_sparse(row)

    if discrete_type is not None:
        row['temporal_pattern'] = discrete_type
        row['spectral'] = get_spectral_for_discrete(discrete_type)
        row['is_discrete_sparse'] = True

        # Apply engine adjustments
        engines = row.get('engines', [])
        adjustments = get_engines_for_discrete(discrete_type)

        for eng in adjustments.get('remove', []):
            if eng == '*':
                engines = []
                break
            if eng in engines:
                engines.remove(eng)

        for eng in adjustments.get('add', []):
            if eng not in engines:
                engines.append(eng)

        row['engines'] = engines
    else:
        row['is_discrete_sparse'] = False

    return row
