"""
Tests for PR8: Robust CONSTANT Detection Fix
"""

import pytest
import sys
sys.path.insert(0, '/Users/jasonrudder/rudder')

from framework.typology.constant_detection import (
    is_constant_signal,
    classify_constant_from_row,
    validate_constant_detection,
    CONSTANT_CONFIG,
)


# ============================================================
# Test: True CONSTANT cases (should return True)
# ============================================================

class TestTrueConstant:
    
    def test_zero_std(self):
        """Zero standard deviation = constant."""
        result = is_constant_signal(
            signal_std=0.0,
            signal_mean=50.0,
            unique_ratio=0.001,
            n_samples=1000,
        )
        assert result is True
    
    def test_tiny_std(self):
        """Essentially zero std = constant."""
        result = is_constant_signal(
            signal_std=1e-12,
            signal_mean=100.0,
            unique_ratio=0.5,
            n_samples=1000,
        )
        assert result is True
    
    def test_tiny_cv_large_scale(self):
        """Large mean with tiny relative variation = constant."""
        result = is_constant_signal(
            signal_std=0.0001,
            signal_mean=1000000.0,
            unique_ratio=0.5,
            n_samples=1000,
        )
        # CV = 0.0001 / 1000000 = 1e-10 < 1e-6
        assert result is True
    
    def test_all_zeros(self):
        """All zeros (Mn_II case) = constant."""
        result = is_constant_signal(
            signal_std=0.0,
            signal_mean=0.0,
            unique_ratio=0.0,
            n_samples=500,
        )
        assert result is True
    
    def test_single_unique_value(self):
        """Only one unique value = constant."""
        result = is_constant_signal(
            signal_std=0.0,
            signal_mean=42.0,
            unique_ratio=1/10000,  # 1 unique in 10000
            n_samples=10000,
        )
        assert result is True


# ============================================================
# Test: NOT CONSTANT cases (should return False)
# ============================================================

class TestNotConstant:
    
    def test_skab_accelerometer1(self):
        """SKAB Accelerometer1RMS - real variation."""
        result = is_constant_signal(
            signal_std=0.00474,
            signal_mean=0.2126,
            unique_ratio=0.738,
            n_samples=9405,
        )
        # CV = 0.00474 / 0.2126 = 0.022 = 2.2% variation
        assert result is False
    
    def test_skab_accelerometer2(self):
        """SKAB Accelerometer2RMS - real variation."""
        result = is_constant_signal(
            signal_std=0.00399,
            signal_mean=0.2684,
            unique_ratio=0.709,
            n_samples=9405,
        )
        assert result is False
    
    def test_skab_temperature(self):
        """SKAB Temperature - definitely not constant."""
        result = is_constant_signal(
            signal_std=0.667,
            signal_mean=89.47,
            unique_ratio=0.812,
            n_samples=9405,
        )
        # CV = 0.667 / 89.47 = 0.0075 = 0.75% variation
        assert result is False
    
    def test_skab_thermocouple(self):
        """SKAB Thermocouple - not constant."""
        result = is_constant_signal(
            signal_std=0.731,
            signal_mean=28.47,
            unique_ratio=0.672,
            n_samples=9405,
        )
        # CV = 0.731 / 28.47 = 0.026 = 2.6% variation
        assert result is False
    
    def test_skab_volume_flow(self):
        """SKAB Volume Flow RateRMS - not constant."""
        result = is_constant_signal(
            signal_std=1.605,
            signal_mean=125.24,
            unique_ratio=0.031,
            n_samples=9405,
        )
        # CV = 1.605 / 125.24 = 0.013 = 1.3% variation
        assert result is False
    
    def test_small_signal_real_variation(self):
        """Small signal with 10% relative variation - NOT constant."""
        result = is_constant_signal(
            signal_std=0.001,
            signal_mean=0.01,
            unique_ratio=0.5,
            n_samples=1000,
        )
        # CV = 0.001 / 0.01 = 0.1 = 10% variation
        assert result is False
    
    def test_low_unique_high_cv(self):
        """Low unique ratio but high CV = NOT constant (discrete signal)."""
        result = is_constant_signal(
            signal_std=1.0,
            signal_mean=5.0,
            unique_ratio=0.0005,  # Only 0.05% unique
            n_samples=10000,
        )
        # CV = 1.0 / 5.0 = 0.2 = 20% variation
        # Low unique ratio suggests discrete, but high CV means real variation
        assert result is False
    
    def test_vix_not_constant(self):
        """VIX - high persistence but real variation."""
        result = is_constant_signal(
            signal_std=7.77,
            signal_mean=19.44,
            unique_ratio=0.277,
            n_samples=9113,
        )
        # CV = 7.77 / 19.44 = 0.4 = 40% variation
        assert result is False


# ============================================================
# Test: Edge cases
# ============================================================

class TestEdgeCases:
    
    def test_nan_std_returns_false(self):
        """NaN std returns False (not constant)."""
        import math
        result = is_constant_signal(
            signal_std=math.nan,
            signal_mean=100.0,
            unique_ratio=0.5,
            n_samples=1000,
        )
        assert result is False
    
    def test_none_std_returns_false(self):
        """None std returns False (not constant)."""
        result = is_constant_signal(
            signal_std=None,
            signal_mean=100.0,
            unique_ratio=0.5,
            n_samples=1000,
        )
        assert result is False
    
    def test_zero_mean_with_variation(self):
        """Zero mean but non-zero std = not constant."""
        result = is_constant_signal(
            signal_std=1.0,
            signal_mean=0.0,
            unique_ratio=0.9,
            n_samples=1000,
        )
        # Can't compute CV when mean=0, but std > threshold
        assert result is False
    
    def test_too_few_samples(self):
        """Too few samples = can't determine, return False."""
        result = is_constant_signal(
            signal_std=0.0,
            signal_mean=100.0,
            unique_ratio=0.001,
            n_samples=5,  # Less than min_samples
        )
        assert result is False
    
    def test_negative_mean(self):
        """Negative mean handled correctly."""
        result = is_constant_signal(
            signal_std=0.0001,
            signal_mean=-1000000.0,
            unique_ratio=0.001,
            n_samples=1000,
        )
        # CV uses absolute mean
        assert result is True


# ============================================================
# Test: Row-based classification
# ============================================================

class TestRowClassification:
    
    def test_classify_from_row(self):
        """Classify from dict row."""
        row = {
            'signal_id': 'test',
            'signal_std': 0.667,
            'signal_mean': 89.47,
            'unique_ratio': 0.812,
            'n_samples': 9405,
        }
        result = classify_constant_from_row(row)
        assert result is False
    
    def test_classify_missing_fields(self):
        """Handle missing fields gracefully."""
        row = {
            'signal_id': 'test',
            # Missing signal_std, signal_mean
            'unique_ratio': 0.5,
            'n_samples': 1000,
        }
        result = classify_constant_from_row(row)
        # Should return False when data is missing
        assert result is False


# ============================================================
# Test: Validation suite
# ============================================================

class TestValidationSuite:
    
    def test_all_validations_pass(self):
        """All validation cases should pass."""
        results = validate_constant_detection()
        
        for test_name, passed, message in results:
            assert passed, f"{test_name}: {message}"
    
    def test_validation_count(self):
        """Should have multiple validation cases."""
        results = validate_constant_detection()
        assert len(results) >= 8


# ============================================================
# Test: Config values
# ============================================================

class TestConfig:
    
    def test_config_has_required_keys(self):
        """Config has all required keys."""
        required = ['signal_std_max', 'cv_max', 'unique_ratio_max', 'min_samples']
        for key in required:
            assert key in CONSTANT_CONFIG
    
    def test_config_values_sensible(self):
        """Config values are sensible."""
        assert CONSTANT_CONFIG['signal_std_max'] < 0.001
        assert CONSTANT_CONFIG['cv_max'] < 0.01
        assert CONSTANT_CONFIG['unique_ratio_max'] < 0.01
        assert CONSTANT_CONFIG['min_samples'] >= 2
