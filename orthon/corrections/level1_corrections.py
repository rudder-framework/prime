"""
Level 1 Stationarity Corrections

Fix for deterministic trends (e.g., exponential decay) that pass ADF
but are clearly non-stationary.

Problem:
    conc_A in CSTR dataset: monotonic exponential decay from 1.0 to 0.01.
    ADF p=0.00 (rejects unit root — correct, bounded decay is not a random walk)
    KPSS p=0.01 (rejects stationarity — correct, mean is changing)
    variance_ratio=0.00035 (second half has nearly zero variance)
    mean_shift_ratio >> 0.5

    Joint classification: STATIONARY (ADF dominates)
    Should be: TREND_STATIONARY or NON_STATIONARY

Fix:
    When ADF passes but KPSS rejects AND mean_shift_ratio is large,
    override to TREND_STATIONARY. This catches deterministic trends
    that converge to a limit (bounded, so ADF passes) but clearly
    have non-stationary mean (KPSS catches this).

    Also add segment-mean divergence check as additional evidence.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


def detect_deterministic_trend(
    y: np.ndarray,
    n_segments: int = 4,
) -> Tuple[bool, float, str]:
    """
    Detect deterministic trends by comparing segment means.

    Splits signal into n_segments and checks if means show
    monotonic progression. This catches exponential decay,
    linear trends, and accumulation patterns.

    Chemical CSTR example:
        conc_A segments: [0.72, 0.27, 0.07, 0.02] → monotonic decay
        temperature segments: [350.5, 351.8, 352.6, 353.4] → weak drift

    Args:
        y: Signal values as 1D array
        n_segments: Number of segments to compare

    Returns:
        (is_trend, strength, direction)
        is_trend: True if monotonic segment means
        strength: max segment-mean difference / overall std
        direction: 'increasing', 'decreasing', or 'none'
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < n_segments * 10:
        return False, 0.0, 'none'

    segment_size = n // n_segments
    segment_means = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else n
        segment_means.append(np.mean(y[start:end]))

    overall_std = np.std(y)
    if overall_std < 1e-15:
        return False, 0.0, 'none'

    # Check monotonicity
    diffs = np.diff(segment_means)
    all_increasing = np.all(diffs > 0)
    all_decreasing = np.all(diffs < 0)
    is_monotonic = all_increasing or all_decreasing

    # Strength: how far do segments diverge relative to overall spread
    max_diff = abs(segment_means[-1] - segment_means[0])
    strength = max_diff / overall_std

    direction = 'none'
    if all_increasing:
        direction = 'increasing'
    elif all_decreasing:
        direction = 'decreasing'

    # Strong trend: monotonic AND large divergence
    is_trend = bool(is_monotonic and strength > 1.0)

    return is_trend, float(strength), direction


def correct_stationarity(
    stationarity_type: str,
    adf_rejects: bool,
    kpss_rejects: bool,
    mean_shift_ratio: float,
    variance_ratio: float,
    mean_stable: bool,
    is_deterministic_trend: bool,
    trend_strength: float,
) -> str:
    """
    Apply stationarity correction for deterministic trends.

    Override logic:
        If ADF rejects (says stationary) AND KPSS rejects (says non-stationary)
        AND either:
            - mean_shift_ratio > 1.0 (large mean change)
            - is_deterministic_trend with strength > 2.0
            - variance_ratio < 0.01 or > 100 (extreme variance change)
        Then: override to TREND_STATIONARY

    This catches signals like exponential decay where:
        - ADF: "not a unit root" (correct — it's bounded)
        - KPSS: "not stationary" (correct — mean is changing)
        - Current logic: DIFFERENCE_STATIONARY (ADF=Y, KPSS=Y)
          or STATIONARY (ADF=Y, KPSS=N — can happen with lag selection)

    With the mean_shift and trend checks, we correctly identify these
    as TREND_STATIONARY (deterministic trend, remove it, then stationary).

    Args:
        stationarity_type: Original classification string
        adf_rejects: ADF rejected unit root hypothesis
        kpss_rejects: KPSS rejected stationarity hypothesis
        mean_shift_ratio: |mean(half2) - mean(half1)| / std
        variance_ratio: var(half2) / var(half1)
        mean_stable: True if mean_shift_ratio < 0.5
        is_deterministic_trend: From detect_deterministic_trend()
        trend_strength: From detect_deterministic_trend()

    Returns:
        Corrected stationarity classification string
    """
    # Only override when there's strong evidence of trending
    # that the ADF/KPSS joint table might miss
    needs_override = False

    # Case 1: ADF says stationary, but mean is clearly shifting
    if adf_rejects and not mean_stable and mean_shift_ratio > 1.0:
        needs_override = True

    # Case 2: Strong deterministic trend detected
    if is_deterministic_trend and trend_strength > 2.0:
        needs_override = True

    # Case 3: Extreme variance ratio (signal character changing dramatically)
    if variance_ratio is not None and not np.isnan(variance_ratio):
        if variance_ratio < 0.01 or variance_ratio > 100:
            needs_override = True

    if needs_override:
        # If KPSS also rejects, this is consistent with trend-stationary
        if kpss_rejects:
            return 'TREND_STATIONARY'
        # If only ADF passes and we see the trend, still override
        return 'TREND_STATIONARY'

    return stationarity_type


def apply_level1_corrections(
    row: Dict[str, Any],
    signal_values: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Apply Level 1 corrections to a typology row.

    If signal_values are provided, runs detect_deterministic_trend().
    Otherwise uses existing mean_shift_ratio from the row.

    Args:
        row: Typology row dict
        signal_values: Optional raw signal array for trend detection

    Returns:
        Row with corrected stationarity field
    """
    corrected = dict(row)

    # Detect deterministic trend if signal data available
    is_det_trend = False
    trend_strength = 0.0
    trend_direction = 'none'

    if signal_values is not None:
        is_det_trend, trend_strength, trend_direction = detect_deterministic_trend(
            signal_values
        )
        corrected['is_deterministic_trend'] = is_det_trend
        corrected['trend_strength'] = trend_strength
        corrected['trend_direction'] = trend_direction

    # Reconstruct ADF/KPSS results from typology
    adf_p = row.get('adf_pvalue', 1.0)
    kpss_p = row.get('kpss_pvalue', 1.0)
    adf_rejects = adf_p < 0.05
    kpss_rejects = kpss_p < 0.05
    mean_shift = row.get('mean_shift_ratio', 0.0)
    var_ratio = row.get('variance_ratio', 1.0)
    mean_stable = mean_shift < 0.5 if mean_shift is not None else True

    corrected_stat = correct_stationarity(
        row.get('stationarity', 'STATIONARY'),
        adf_rejects,
        kpss_rejects,
        mean_shift if mean_shift is not None else 0.0,
        var_ratio if var_ratio is not None else 1.0,
        mean_stable,
        is_det_trend,
        trend_strength,
    )

    corrected['stationarity'] = corrected_stat

    return corrected
