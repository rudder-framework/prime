"""
Tests for classification stability checker.

Run with: pytest test_classification_stability.py -v
"""

import numpy as np
import pytest
from prime.typology.classification_stability import (
    check_classification_stability,
    compare_classifications,
    extract_window,
    StabilityStatus,
    StabilityResult,
)


# ============================================================
# Mock classify functions for testing
# ============================================================

def mock_classify_stable(data: np.ndarray) -> dict:
    """Always returns same classification - simulates stable signal."""
    return {
        "signal_type": "periodic_trend",
        "dominant_component": "trend",
        "trend_strength": 0.7,
        "periodicity_strength": 0.5,
        "noise_ratio": 0.1,
        "confidence": 0.85
    }


def mock_classify_by_mean(data: np.ndarray) -> dict:
    """
    Classification changes based on data mean.
    Simulates regime change detection.
    """
    mean = np.mean(data)
    
    if mean < 0.3:
        return {
            "signal_type": "noise_dominant",
            "dominant_component": "noise",
            "trend_strength": 0.1,
            "periodicity_strength": 0.2,
            "noise_ratio": 0.8,
            "confidence": 0.75
        }
    elif mean < 0.7:
        return {
            "signal_type": "periodic_trend",
            "dominant_component": "trend",
            "trend_strength": 0.6,
            "periodicity_strength": 0.5,
            "noise_ratio": 0.2,
            "confidence": 0.80
        }
    else:
        return {
            "signal_type": "strong_trend",
            "dominant_component": "trend",
            "trend_strength": 0.9,
            "periodicity_strength": 0.3,
            "noise_ratio": 0.1,
            "confidence": 0.90
        }


# ============================================================
# Test: extract_window
# ============================================================

def test_extract_window_basic():
    """Window extraction returns correct slice."""
    data = np.arange(100)
    window = extract_window(data, start=10, size=20)
    
    assert len(window) == 20
    assert window[0] == 10
    assert window[-1] == 29


def test_extract_window_end_boundary():
    """Window extraction handles end of array."""
    data = np.arange(100)
    window = extract_window(data, start=90, size=20)
    
    # Should only get 10 rows (90-99)
    assert len(window) == 10
    assert window[-1] == 99


# ============================================================
# Test: compare_classifications
# ============================================================

def test_compare_identical():
    """Identical classifications should match."""
    c1 = {"signal_type": "trend", "trend_strength": 0.5}
    c2 = {"signal_type": "trend", "trend_strength": 0.5}
    
    match, diff = compare_classifications(c1, c2)
    
    assert match is True
    assert diff is None


def test_compare_type_difference():
    """Different signal types should not match."""
    c1 = {"signal_type": "trend", "trend_strength": 0.5}
    c2 = {"signal_type": "noise", "trend_strength": 0.5}
    
    match, diff = compare_classifications(c1, c2)
    
    assert match is False
    assert "signal_type" in diff


def test_compare_within_tolerance():
    """Numeric values within tolerance should match."""
    c1 = {"signal_type": "trend", "trend_strength": 0.50}
    c2 = {"signal_type": "trend", "trend_strength": 0.55}
    
    match, diff = compare_classifications(c1, c2, tolerance=0.1)
    
    assert match is True


def test_compare_outside_tolerance():
    """Numeric values outside tolerance should not match."""
    c1 = {"signal_type": "trend", "trend_strength": 0.50}
    c2 = {"signal_type": "trend", "trend_strength": 0.75}
    
    match, diff = compare_classifications(c1, c2, tolerance=0.1)
    
    assert match is False
    assert "trend_strength" in diff


# ============================================================
# Test: check_classification_stability
# ============================================================

def test_stability_stable_signal():
    """Stable signal should return STABLE status."""
    # Create uniform data - same throughout
    data = np.random.uniform(0.4, 0.6, size=50000)
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_stable,
        window_size=10000
    )
    
    assert result.status == StabilityStatus.STABLE
    assert result.match is True
    assert result.differences is None


def test_stability_regime_change():
    """Signal with regime change should return REGIME_CHANGE status."""
    # Create data with different regimes: low mean -> high mean
    first_half = np.random.uniform(0.1, 0.2, size=25000)  # Low mean
    second_half = np.random.uniform(0.8, 0.9, size=25000)  # High mean
    data = np.concatenate([first_half, second_half])
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_by_mean,
        window_size=10000
    )
    
    assert result.status == StabilityStatus.REGIME_CHANGE
    assert result.match is False
    assert result.differences is not None
    assert "signal_type" in result.differences


def test_stability_insufficient_data():
    """Insufficient data should return INSUFFICIENT_DATA status."""
    data = np.random.uniform(0, 1, size=5000)  # Less than 2 windows
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_stable,
        window_size=10000
    )
    
    assert result.status == StabilityStatus.INSUFFICIENT_DATA


def test_stability_window_positions():
    """Verify correct window positions are captured."""
    data = np.random.uniform(0, 1, size=56000000)  # 56M rows like industrial
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_stable,
        window_size=10000
    )
    
    # First window: 0 to 10000
    assert result.primary_classification.start_row == 0
    assert result.primary_classification.end_row == 10000
    
    # Last window: 55990000 to 56000000
    assert result.verification_classification.start_row == 56000000 - 10000
    assert result.verification_classification.end_row == 56000000


# ============================================================
# Test: Real-world scenarios
# ============================================================

def test_gradual_drift():
    """Gradual drift might stay within tolerance."""
    # Signal that drifts gradually
    data = np.linspace(0.4, 0.6, 50000)  # Slight upward drift
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_by_mean,
        window_size=10000,
        tolerance=0.15  # Wider tolerance
    )
    
    # Might be stable or might catch the drift
    # depends on tolerance setting
    assert result.status in [StabilityStatus.STABLE, StabilityStatus.REGIME_CHANGE]


def test_sudden_regime_shift_mid_signal():
    """
    Regime shift in middle should be caught.
    First and last windows will differ.
    """
    # Normal -> Crisis -> Normal pattern
    part1 = np.random.uniform(0.4, 0.6, size=20000)   # Normal
    part2 = np.random.uniform(0.0, 0.1, size=20000)   # Crisis (won't be in windows)
    part3 = np.random.uniform(0.8, 0.9, size=20000)   # New regime
    
    data = np.concatenate([part1, part2, part3])
    
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_by_mean,
        window_size=10000
    )
    
    # First window (normal) vs last window (new regime) should differ
    assert result.status == StabilityStatus.REGIME_CHANGE


# ============================================================
# Test: SQL output format
# ============================================================

def test_stability_result_to_record():
    """Verify SQL-friendly output format."""
    from prime.typology.classification_stability import stability_result_to_record
    
    data = np.random.uniform(0, 1, size=50000)
    result = check_classification_stability(
        data,
        classify_fn=mock_classify_stable,
        window_size=10000
    )
    
    record = stability_result_to_record(result)
    
    # Should be flat dict suitable for SQL
    assert isinstance(record, dict)
    assert "status" in record
    assert "match" in record
    assert "primary_window_start" in record
    assert "verification_window_start" in record
    
    # Values should be SQL-friendly types
    assert isinstance(record["status"], str)
    assert isinstance(record["match"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
