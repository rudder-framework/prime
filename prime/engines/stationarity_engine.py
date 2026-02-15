"""
Level 1: Stationarity Engine

Determine if signal statistics are stable over time.

Tests:
- KPSS: H0 = stationary, H1 = unit root
- ADF: H0 = unit root, H1 = stationary

Combined interpretation:
- KPSS fail to reject + ADF reject = STATIONARY
- KPSS reject + ADF fail to reject = NON_STATIONARY
- Both reject = TREND_STATIONARY
- Neither reject = INCONCLUSIVE
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings


class StationarityClass(Enum):
    """Stationarity classification."""
    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    TREND_STATIONARY = "trend_stationary"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StationarityResult:
    """Result of stationarity testing."""
    classification: StationarityClass
    kpss_statistic: Optional[float]
    kpss_pvalue: Optional[float]
    adf_statistic: Optional[float]
    adf_pvalue: Optional[float]
    confidence: float
    description: str


def test_stationarity(
    values: np.ndarray,
    significance: float = 0.05,
) -> StationarityResult:
    """
    Test signal for stationarity using KPSS and ADF tests.

    Args:
        values: Time series values
        significance: Significance level (default 0.05)

    Returns:
        StationarityResult with classification and test statistics
    """
    n = len(values)

    if n < 20:
        return StationarityResult(
            classification=StationarityClass.INCONCLUSIVE,
            kpss_statistic=None,
            kpss_pvalue=None,
            adf_statistic=None,
            adf_pvalue=None,
            confidence=0.0,
            description="Insufficient data for stationarity testing",
        )

    # Check for constant signal
    if np.std(values) < 1e-10:
        return StationarityResult(
            classification=StationarityClass.STATIONARY,
            kpss_statistic=0.0,
            kpss_pvalue=1.0,
            adf_statistic=-np.inf,
            adf_pvalue=0.0,
            confidence=1.0,
            description="Constant signal - trivially stationary",
        )

    kpss_stat = None
    kpss_pvalue = None
    adf_stat = None
    adf_pvalue = None

    try:
        from primitives.stat_tests.stationarity_tests import kpss_test, adf_test

        # KPSS test (H0: stationary)
        kpss_result = kpss_test(values, regression='c', nlags='auto')
        kpss_stat = float(kpss_result[0])
        kpss_pvalue = float(kpss_result[1])

        # ADF test (H0: unit root)
        adf_result = adf_test(values, regression='c')
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])

    except Exception as e:
        return StationarityResult(
            classification=StationarityClass.INCONCLUSIVE,
            kpss_statistic=kpss_stat,
            kpss_pvalue=kpss_pvalue,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            confidence=0.0,
            description=f"Stationarity test error: {str(e)}",
        )

    # Classification based on test results
    kpss_reject = kpss_pvalue < significance
    adf_reject = adf_pvalue < significance

    if not kpss_reject and adf_reject:
        classification = StationarityClass.STATIONARY
        confidence = min(kpss_pvalue, 1 - adf_pvalue)
        description = "Signal is stationary (stable statistics over time)"

    elif kpss_reject and not adf_reject:
        classification = StationarityClass.NON_STATIONARY
        confidence = min(1 - kpss_pvalue, adf_pvalue)
        description = "Signal is non-stationary (evolving statistics)"

    elif kpss_reject and adf_reject:
        classification = StationarityClass.TREND_STATIONARY
        confidence = min(1 - kpss_pvalue, 1 - adf_pvalue)
        description = "Signal is trend-stationary (stationary around trend)"

    else:
        classification = StationarityClass.INCONCLUSIVE
        confidence = 0.5
        description = "Stationarity tests inconclusive"

    return StationarityResult(
        classification=classification,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_pvalue,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        confidence=float(np.clip(confidence, 0, 1)),
        description=description,
    )


def compute_stationarity_metrics(values: np.ndarray) -> dict:
    """
    Compute additional stationarity-related metrics.

    Args:
        values: Time series values

    Returns:
        Dict with metrics: rolling_mean_drift, rolling_var_drift, etc.
    """
    n = len(values)
    if n < 10:
        return {
            'rolling_mean_drift': 0.0,
            'rolling_var_drift': 0.0,
            'mean_reversion_rate': 0.0,
        }

    # Split into halves
    half = n // 2
    first_half = values[:half]
    second_half = values[half:]

    # Mean drift
    mean_drift = abs(np.mean(second_half) - np.mean(first_half))
    mean_drift_norm = mean_drift / (np.std(values) + 1e-10)

    # Variance drift
    var_first = np.var(first_half)
    var_second = np.var(second_half)
    var_drift = abs(var_second - var_first) / (var_first + 1e-10)

    # Mean reversion rate (from AR1 coefficient)
    if np.std(values) > 1e-10 and n > 1:
        ar1 = np.corrcoef(values[:-1], values[1:])[0, 1]
        mean_reversion_rate = 1 - abs(ar1) if np.isfinite(ar1) else 0.0
    else:
        mean_reversion_rate = 0.0

    return {
        'rolling_mean_drift': float(mean_drift_norm),
        'rolling_var_drift': float(var_drift),
        'mean_reversion_rate': float(mean_reversion_rate),
    }
