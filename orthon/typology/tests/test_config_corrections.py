"""
Tests for config-driven typology corrections.

Validates that:
1. Config values are accessible
2. Classifications match expected behavior
3. Config changes propagate correctly
"""

import pytest
import math
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orthon.config.typology_config import (
    TYPOLOGY_CONFIG,
    get_threshold,
    validate_config,
)
from orthon.typology.level2_corrections import (
    is_first_bin_artifact,
    is_genuine_periodic,
    classify_temporal_pattern,
    classify_spectral,
    correct_engines,
    apply_corrections,
)


# ============================================================
# Config Access Tests
# ============================================================

class TestConfigAccess:
    
    def test_get_threshold_simple(self):
        """Can access top-level config."""
        fft = get_threshold('artifacts.default_fft_size')
        assert fft == 256
    
    def test_get_threshold_nested(self):
        """Can access deeply nested config."""
        hurst = get_threshold('temporal.trending.hurst_strong')
        assert hurst == 0.99
    
    def test_get_threshold_default(self):
        """Returns default for missing keys."""
        val = get_threshold('nonexistent.path', default=42)
        assert val == 42
    
    def test_config_validation(self):
        """Config passes internal consistency checks."""
        errors = validate_config()
        assert len(errors) == 0, f"Config errors: {errors}"


# ============================================================
# Artifact Detection Tests
# ============================================================

class TestArtifactDetection:
    
    def test_first_bin_detected(self):
        """First-bin artifact is detected."""
        # 1/256 = 0.00390625
        assert is_first_bin_artifact(0.00390625, 5000, 256, -0.5) is True
    
    def test_first_bin_needs_red_slope(self):
        """First-bin needs negative slope to be artifact."""
        assert is_first_bin_artifact(0.00390625, 5000, 256, 0.1) is False
    
    def test_real_peak_not_artifact(self):
        """Real spectral peak not flagged."""
        assert is_first_bin_artifact(0.125, 5000, 256, -0.5) is False
    
    def test_none_frequency(self):
        """None frequency returns False."""
        assert is_first_bin_artifact(None, 5000, 256, -0.5) is False


# ============================================================
# Temporal Pattern Tests
# ============================================================

class TestTemporalPattern:
    
    def test_strong_hurst_is_trending(self):
        """hurst >= 0.99 → TRENDING."""
        row = {'hurst': 0.995, 'n_samples': 200}
        assert classify_temporal_pattern(row) == 'TRENDING'
    
    def test_moderate_hurst_acf_nan_trending(self):
        """hurst > 0.85 + acf=NaN + low entropy → TRENDING."""
        row = {
            'hurst': 0.90,
            'acf_half_life': None,
            'sample_entropy': 0.05,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) == 'TRENDING'
    
    def test_moderate_hurst_long_acf_trending(self):
        """hurst > 0.85 + long relative ACF + low entropy → TRENDING."""
        row = {
            'hurst': 0.90,
            'acf_half_life': 30,  # 30/200 = 0.15 > 0.10
            'sample_entropy': 0.10,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) == 'TRENDING'
    
    def test_flat_high_entropy_is_random(self):
        """Flat spectrum + high entropy → RANDOM."""
        row = {
            'spectral_flatness': 0.95,
            'perm_entropy': 0.995,
            'hurst': 0.5,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) == 'RANDOM'
    
    def test_chaotic_needs_long_series(self):
        """CHAOTIC requires n >= 500."""
        short = {
            'lyapunov_proxy': 0.8,
            'perm_entropy': 0.98,
            'hurst': 0.5,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(short) != 'CHAOTIC'
        
        long = dict(short)
        long['n_samples'] = 1000
        assert classify_temporal_pattern(long) == 'CHAOTIC'
    
    def test_low_tpr_is_quasi_periodic(self):
        """Low turning point ratio → QUASI_PERIODIC."""
        row = {
            'turning_point_ratio': 0.5,
            'hurst': 0.6,
            'spectral_flatness': 0.5,
            'perm_entropy': 0.8,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) == 'QUASI_PERIODIC'
    
    def test_default_is_stationary(self):
        """Nothing matches → STATIONARY."""
        row = {
            'hurst': 0.6,
            'turning_point_ratio': 0.8,
            'spectral_flatness': 0.5,
            'perm_entropy': 0.8,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) == 'STATIONARY'

    def test_zero_std_is_constant(self):
        """signal_std == 0 → CONSTANT (all-zeros like Mn_II)."""
        row = {
            'signal_std': 0.0,
            'signal_mean': 0.0,
            'hurst': 0.5,
            'n_samples': 1000,
        }
        assert classify_temporal_pattern(row) == 'CONSTANT'

    def test_very_low_variance_is_constant(self):
        """Very low variance relative to mean → CONSTANT."""
        row = {
            'signal_std': 0.001,
            'signal_mean': 100.0,  # variance_ratio = 0.001^2 / 100^2 = 1e-10
            'hurst': 0.5,
            'n_samples': 1000,
        }
        assert classify_temporal_pattern(row) == 'CONSTANT'

    def test_normal_variance_not_constant(self):
        """Normal variance should NOT be CONSTANT."""
        row = {
            'signal_std': 10.0,
            'signal_mean': 100.0,  # variance_ratio = 0.01 > 0.001
            'hurst': 0.6,
            'turning_point_ratio': 0.8,
            'n_samples': 200,
        }
        assert classify_temporal_pattern(row) != 'CONSTANT'


# ============================================================
# Spectral Classification Tests
# ============================================================

class TestSpectralClassification:
    
    def test_periodic_high_hnr_is_harmonic(self):
        """PERIODIC + high HNR → HARMONIC."""
        row = {'harmonic_noise_ratio': 5.0}
        assert classify_spectral(row, 'PERIODIC') == 'HARMONIC'
    
    def test_periodic_low_hnr_is_narrowband(self):
        """PERIODIC + low HNR → NARROWBAND."""
        row = {'harmonic_noise_ratio': 1.0}
        assert classify_spectral(row, 'PERIODIC') == 'NARROWBAND'
    
    def test_flat_spectrum_is_broadband(self):
        """Flat spectrum → BROADBAND."""
        row = {'spectral_flatness': 0.9}
        assert classify_spectral(row, 'RANDOM') == 'BROADBAND'
    
    def test_negative_slope_is_red_noise(self):
        """Negative slope → RED_NOISE."""
        row = {'spectral_flatness': 0.3, 'spectral_slope': -0.7}
        assert classify_spectral(row, 'TRENDING') == 'RED_NOISE'
    
    def test_positive_slope_is_blue_noise(self):
        """Positive slope → BLUE_NOISE."""
        row = {'spectral_flatness': 0.3, 'spectral_slope': 0.5}
        assert classify_spectral(row, 'STATIONARY') == 'BLUE_NOISE'


# ============================================================
# Engine Correction Tests
# ============================================================

class TestEngineCorrections:
    
    def test_trending_removes_harmonic_engines(self):
        """TRENDING removes harmonic engines."""
        engines = ['kurtosis', 'harmonics_ratio', 'thd']
        corrected = correct_engines(engines, 'TRENDING', 'RED_NOISE')
        assert 'harmonics_ratio' not in corrected
        assert 'thd' not in corrected
        assert 'kurtosis' in corrected
    
    def test_trending_adds_trend_engines(self):
        """TRENDING adds trend engines."""
        engines = ['kurtosis']
        corrected = correct_engines(engines, 'TRENDING', 'RED_NOISE')
        assert 'hurst' in corrected
        assert 'trend_r2' in corrected
    
    def test_red_noise_adds_psd_slope(self):
        """RED_NOISE spectral adds psd_slope."""
        engines = ['kurtosis']
        corrected = correct_engines(engines, 'STATIONARY', 'RED_NOISE')
        assert 'psd_slope' in corrected


# ============================================================
# Integration Tests: Real Dataset Rows
# ============================================================

# CSTR exponential decay
CSTR_CONC_A = {
    'signal_id': 'conc_A',
    'n_samples': 5000,
    'dominant_frequency': 0.00390625,  # First bin
    'acf_half_life': None,
    'turning_point_ratio': 0.959,
    'spectral_peak_snr': 4.2,
    'spectral_flatness': 0.30,
    'spectral_slope': -0.57,
    'harmonic_noise_ratio': 0.5,
    'hurst': 1.0,
    'perm_entropy': 0.7,
    'sample_entropy': 0.01,
    'lyapunov_proxy': 0.1,
    'engines': ['kurtosis', 'harmonics_ratio'],
    'variance_ratio': 5.0,  # Expanding variance (true trend)
}

# C-MAPSS noisy degradation (full dataset version)
# Note: Real noisy trends have low SNR (noise masks any spectral peak)
CMAPSS_SENSOR_02_NOISY = {
    'signal_id': 'sensor_02',
    'n_samples': 200,
    'dominant_frequency': 0.025,
    'acf_half_life': 35,  # 35/200 = 0.175 > 0.10
    'turning_point_ratio': 0.72,
    'spectral_peak_snr': 4.0,   # Low SNR - trend, not periodic
    'spectral_flatness': 0.35,
    'spectral_slope': -1.2,
    'harmonic_noise_ratio': 2.0,  # Low HNR
    'hurst': 0.88,  # Not quite 0.99
    'perm_entropy': 0.75,
    'sample_entropy': 0.08,
    'lyapunov_proxy': 0.2,
    'variance_ratio': 2.0,
}

# Bearing vibration
BEARING_ACC_X = {
    'signal_id': 'acc_x',
    'n_samples': 10000,
    'dominant_frequency': 0.03125,
    'acf_half_life': 5.0,
    'turning_point_ratio': 0.633,
    'spectral_peak_snr': 31.0,
    'spectral_flatness': 0.068,
    'spectral_slope': -0.58,
    'harmonic_noise_ratio': 1.0,
    'hurst': 0.61,
    'perm_entropy': 0.91,
    'sample_entropy': 0.80,
    'lyapunov_proxy': 1.5,
    'variance_ratio': 1.2,
}

# NEW: Guitar A3 - noisy periodic with high TPR
GUITAR_A3 = {
    'signal_id': 'velocity_mms',
    'n_samples': 5000,
    'dominant_frequency': 0.441406,  # ~220 Hz (A3)
    'acf_half_life': 1.0,
    'turning_point_ratio': 1.32,     # > 0.95 normally fails
    'spectral_peak_snr': 52.6,       # Very high SNR
    'spectral_flatness': 0.001,      # Very concentrated
    'spectral_slope': 0.66,
    'harmonic_noise_ratio': 0.76,
    'hurst': 0.008,
    'perm_entropy': 0.43,
    'sample_entropy': 1.5,
    'lyapunov_proxy': 0.3,
    'variance_ratio': 1.0,
}

# NEW: Battery degradation - oscillating trend
BATTERY_CAPACITY = {
    'signal_id': 'charge_capacity',
    'n_samples': 781,
    'dominant_frequency': 0.05,
    'acf_half_life': 10.0,
    'turning_point_ratio': 0.50,
    'spectral_peak_snr': 8.0,
    'spectral_flatness': 0.98,
    'spectral_slope': -0.1,
    'harmonic_noise_ratio': 0.5,
    'hurst': 0.70,                   # Not high enough for simple TRENDING gate
    'perm_entropy': 0.79,
    'sample_entropy': 0.5,
    'lyapunov_proxy': 0.76,
    'variance_ratio': 1.5,
    'segment_means': [1.10, 0.95, 0.75, 0.55],  # Monotonic decline
    'signal_mean': 0.84,
}

# NEW: Double pendulum - smooth bounded chaos
DOUBLE_PENDULUM_THETA1 = {
    'signal_id': 'theta1',
    'n_samples': 10000,
    'dominant_frequency': 0.01,
    'acf_half_life': 80.0,
    'turning_point_ratio': 0.5,
    'spectral_peak_snr': 5.0,
    'spectral_flatness': 0.15,
    'spectral_slope': -5.83,
    'harmonic_noise_ratio': 0.3,
    'hurst': 1.0,                    # Very high (smooth trajectory)
    'perm_entropy': 0.41,            # Low (locally predictable)
    'sample_entropy': 0.04,
    'lyapunov_proxy': 0.19,
    'variance_ratio': 1.5,           # Bounded (not expanding)
}


class TestIntegration:
    
    def test_cstr_conc_a_becomes_trending(self):
        """CSTR conc_A: false PERIODIC → TRENDING."""
        corrected = apply_corrections(CSTR_CONC_A)
        assert corrected['temporal_pattern'] == 'TRENDING'
        assert corrected['spectral'] == 'RED_NOISE'
        assert corrected['dominant_frequency_is_artifact'] is True
    
    def test_cmapss_noisy_trend_becomes_trending(self):
        """C-MAPSS noisy sensor with hurst=0.88 → TRENDING (via relative ACF)."""
        corrected = apply_corrections(CMAPSS_SENSOR_02_NOISY)
        # hurst=0.88 > 0.85, acf_ratio=0.175 > 0.10, se=0.08 < 0.15
        assert corrected['temporal_pattern'] == 'TRENDING'
    
    def test_bearing_becomes_periodic(self):
        """Bearing vibration → PERIODIC."""
        corrected = apply_corrections(BEARING_ACC_X)
        assert corrected['temporal_pattern'] == 'PERIODIC'
        assert corrected['spectral'] == 'NARROWBAND'
    
    def test_cross_dataset_differentiation(self):
        """Different domains get different classifications."""
        cstr = apply_corrections(CSTR_CONC_A)
        cmapss = apply_corrections(CMAPSS_SENSOR_02_NOISY)
        bearing = apply_corrections(BEARING_ACC_X)
        
        # All three should be distinct OR at least CSTR/CMAPSS both trending
        assert cstr['temporal_pattern'] == 'TRENDING'
        assert cmapss['temporal_pattern'] == 'TRENDING'
        assert bearing['temporal_pattern'] == 'PERIODIC'
    
    def test_guitar_spectral_override(self):
        """Guitar A3: high SNR + low flatness → PERIODIC despite high TPR."""
        corrected = apply_corrections(GUITAR_A3)
        # SNR=52.6 > 30, flatness=0.001 < 0.1 → spectral override
        assert corrected['temporal_pattern'] == 'PERIODIC'
    
    def test_battery_segment_trend(self):
        """Battery: oscillating trend → TRENDING via segment analysis."""
        corrected = apply_corrections(BATTERY_CAPACITY)
        # segment_means show monotonic 55% decline
        assert corrected['temporal_pattern'] == 'TRENDING'
    
    def test_pendulum_bounded_deterministic(self):
        """Double pendulum: smooth bounded chaos → NOT TRENDING."""
        corrected = apply_corrections(DOUBLE_PENDULUM_THETA1)
        # hurst=1.0 but variance bounded → should NOT be TRENDING
        # Should be QUASI_PERIODIC or STATIONARY (bounded smooth motion)
        assert corrected['temporal_pattern'] != 'TRENDING'


# ============================================================
# Config Override Tests
# ============================================================

class TestConfigOverride:
    """Test that changing config values affects classification."""
    
    def test_hurst_threshold_affects_trending(self):
        """Changing hurst_strong changes what becomes TRENDING."""
        row = {'hurst': 0.97, 'n_samples': 200}
        
        # With default (0.99), 0.97 is NOT trending
        original = TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong']
        assert classify_temporal_pattern(row) != 'TRENDING'
        
        # Temporarily lower threshold
        TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong'] = 0.95
        try:
            assert classify_temporal_pattern(row) == 'TRENDING'
        finally:
            # Restore
            TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong'] = original


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
