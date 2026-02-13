"""
Tests for PR9: Window/Stride from Characteristic Time
"""

import pytest
import math
import sys
sys.path.insert(0, '/Users/jasonrudder/rudder')

from framework.manifest.characteristic_time import (
    compute_characteristic_time,
    classify_dynamics_speed,
    compute_window_stride,
    compute_window_config,
    WINDOW_CONFIG,
)


# ============================================================
# Characteristic Time: Source Priority
# ============================================================

class TestCharacteristicTimeSource:
    
    def test_acf_half_life_primary(self):
        """ACF half-life is highest priority."""
        char_time, source = compute_characteristic_time(
            acf_half_life=50.0,
            dominant_frequency=0.01,  # Period=100
            turning_point_ratio=0.67,
            hurst=0.9,
            n_samples=1000,
        )
        assert source == 'acf_half_life'
        assert char_time == 50.0
    
    def test_dominant_freq_when_no_acf(self):
        """Dominant frequency used when ACF unavailable."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=0.05,  # Period=20
            turning_point_ratio=0.67,
            hurst=0.5,
            n_samples=1000,
        )
        assert source == 'dominant_frequency'
        assert char_time == 20.0
    
    def test_inter_event_time(self):
        """Inter-event time for impulsive/sparse signals."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=None,
            turning_point_ratio=0.67,
            hurst=0.5,
            n_samples=10000,
            inter_event_time=200.0,
        )
        assert source == 'inter_event_time'
        assert char_time == 200.0
    
    def test_derivative_sparsity_for_step(self):
        """Derivative sparsity for step signals."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=None,
            turning_point_ratio=0.1,
            hurst=0.5,
            n_samples=1000,
            derivative_sparsity=0.95,  # 95% zeros = long levels
        )
        assert source == 'derivative_sparsity'
    
    def test_hurst_for_persistent(self):
        """High hurst for trending/persistent signals."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=None,
            turning_point_ratio=None,
            hurst=0.95,
            n_samples=1000,
        )
        assert source == 'hurst'
        assert char_time > 32
    
    def test_fallback(self):
        """Fallback when nothing available."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=None,
            turning_point_ratio=None,
            hurst=None,
            n_samples=10000,
        )
        assert source == 'fallback'
        assert char_time >= 64


# ============================================================
# Characteristic Time: Edge Cases
# ============================================================

class TestCharacteristicTimeEdgeCases:
    
    def test_nan_ignored(self):
        """NaN values ignored."""
        char_time, source = compute_characteristic_time(
            acf_half_life=math.nan,
            dominant_frequency=0.02,
            turning_point_ratio=math.nan,
            hurst=math.nan,
            n_samples=1000,
        )
        assert source == 'dominant_frequency'
    
    def test_zero_ignored(self):
        """Zero values ignored."""
        char_time, source = compute_characteristic_time(
            acf_half_life=0.0,
            dominant_frequency=0.0,
            turning_point_ratio=0.67,
            hurst=0.5,
            n_samples=1000,
        )
        assert source != 'acf_half_life'
        assert source != 'dominant_frequency'
    
    def test_period_exceeds_signal(self):
        """Period > signal length ignored."""
        char_time, source = compute_characteristic_time(
            acf_half_life=None,
            dominant_frequency=0.0001,  # Period=10000
            turning_point_ratio=0.67,
            hurst=0.5,
            n_samples=100,  # Only 100 samples
        )
        assert source != 'dominant_frequency'


# ============================================================
# Dynamics Speed
# ============================================================

class TestDynamicsSpeed:
    
    def test_fast(self):
        """Fast: char_time < 1% of signal."""
        speed = classify_dynamics_speed(10, 10000)
        assert speed == 'fast'
    
    def test_slow(self):
        """Slow: char_time > 10% of signal."""
        speed = classify_dynamics_speed(2000, 10000)
        assert speed == 'slow'
    
    def test_medium(self):
        """Medium: in between."""
        speed = classify_dynamics_speed(500, 10000)
        assert speed == 'medium'


# ============================================================
# Window/Stride Computation
# ============================================================

class TestWindowStride:
    
    def test_basic(self):
        """Basic window/stride computation."""
        ws = compute_window_stride(100, 10000)
        assert ws['window'] == 250  # 100 * 2.5
        assert ws['stride'] > 0
        assert ws['stride'] <= ws['window']
    
    def test_min_window(self):
        """Respects minimum window."""
        ws = compute_window_stride(10, 10000)
        assert ws['window'] >= WINDOW_CONFIG['min_window']
    
    def test_max_window(self):
        """Respects maximum window."""
        ws = compute_window_stride(10000, 100000)
        assert ws['window'] <= WINDOW_CONFIG['max_window']
    
    def test_window_capped_by_signal(self):
        """Window can't exceed signal length."""
        ws = compute_window_stride(500, 200)
        assert ws['window'] <= 200
    
    def test_fast_more_overlap(self):
        """Fast dynamics → more overlap."""
        ws = compute_window_stride(100, 100000, dynamics_speed='fast')
        assert ws['overlap'] >= 0.70
    
    def test_slow_less_overlap(self):
        """Slow dynamics → less overlap."""
        ws = compute_window_stride(100, 100000, dynamics_speed='slow')
        assert ws['overlap'] <= 0.30


# ============================================================
# Full Row Computation
# ============================================================

class TestComputeWindowConfig:
    
    def test_periodic_signal(self):
        """Periodic: window from frequency."""
        row = {
            'signal_id': 'vib',
            'n_samples': 10000,
            'acf_half_life': None,
            'dominant_frequency': 0.01,
            'turning_point_ratio': 0.8,
            'hurst': 0.5,
        }
        cfg = compute_window_config(row)
        
        assert cfg['characteristic_source'] == 'dominant_frequency'
        assert cfg['characteristic_time'] == 100.0
        assert cfg['window_size'] == 250
    
    def test_trending_signal(self):
        """Trending: window from ACF."""
        row = {
            'signal_id': 'temp',
            'n_samples': 5000,
            'acf_half_life': 200.0,
            'dominant_frequency': None,
            'turning_point_ratio': 0.3,
            'hurst': 0.95,
        }
        cfg = compute_window_config(row)
        
        assert cfg['characteristic_source'] == 'acf_half_life'
        assert cfg['window_size'] == 500
    
    def test_step_signal(self):
        """Step: window from derivative sparsity."""
        row = {
            'signal_id': 'valve',
            'n_samples': 10000,
            'acf_half_life': None,
            'dominant_frequency': None,
            'turning_point_ratio': 0.1,
            'hurst': 0.5,
            'derivative_sparsity': 0.98,
        }
        cfg = compute_window_config(row)
        
        assert cfg['characteristic_source'] == 'derivative_sparsity'
    
    def test_output_keys(self):
        """All expected keys present."""
        row = {
            'signal_id': 'test',
            'n_samples': 1000,
            'hurst': 0.5,
        }
        cfg = compute_window_config(row)
        
        expected = ['window_size', 'stride', 'overlap', 
                    'characteristic_time', 'characteristic_source', 'dynamics_speed']
        for key in expected:
            assert key in cfg


# ============================================================
# Real Dataset Cases
# ============================================================

class TestRealDatasets:
    
    def test_skab_accelerometer(self):
        """SKAB accelerometer."""
        row = {
            'signal_id': 'Accelerometer1RMS',
            'n_samples': 9405,
            'acf_half_life': 50.0,
            'dominant_frequency': None,
            'turning_point_ratio': 0.65,
            'hurst': 0.88,
        }
        cfg = compute_window_config(row)
        
        assert cfg['characteristic_source'] == 'acf_half_life'
        assert 64 <= cfg['window_size'] <= 2048
    
    def test_vix(self):
        """VIX: long memory."""
        row = {
            'signal_id': 'vix',
            'n_samples': 9113,
            'acf_half_life': 78.0,
            'dominant_frequency': 0.0,
            'turning_point_ratio': 0.76,
            'hurst': 0.97,
        }
        cfg = compute_window_config(row)
        
        assert cfg['characteristic_source'] == 'acf_half_life'
        assert cfg['window_size'] >= 150
    
    def test_building_vibration_random(self):
        """Building vibration RANDOM signal."""
        row = {
            'signal_id': 'floor_x_sway',
            'n_samples': 10000,
            'acf_half_life': None,
            'dominant_frequency': None,
            'turning_point_ratio': 0.67,
            'hurst': 0.5,
        }
        cfg = compute_window_config(row)
        
        # Should use TPR or fallback
        assert cfg['window_size'] >= 64
    
    def test_building_vibration_trending(self):
        """Building vibration TRENDING signal."""
        row = {
            'signal_id': 'floor_z_vc_e',
            'n_samples': 10000,
            'acf_half_life': None,
            'dominant_frequency': None,
            'turning_point_ratio': 0.75,
            'hurst': 1.0,
        }
        cfg = compute_window_config(row)
        
        # High hurst should drive
        assert cfg['characteristic_source'] == 'hurst'
    
    def test_earthquake_short(self):
        """Earthquake: very short signal."""
        row = {
            'signal_id': 'daily_count',
            'n_samples': 30,
            'acf_half_life': None,
            'dominant_frequency': None,
            'turning_point_ratio': 0.86,
            'hurst': 0.76,
        }
        cfg = compute_window_config(row)
        
        # Window can't exceed signal
        assert cfg['window_size'] <= 30
        assert cfg['stride'] <= cfg['window_size']


# ============================================================
# Config Validation
# ============================================================

class TestConfig:
    
    def test_required_keys(self):
        """Config has required keys."""
        required = ['window_multiplier', 'min_window', 'max_window', 
                    'stride_fraction', 'min_stride', 'dynamics_speed']
        for key in required:
            assert key in WINDOW_CONFIG
    
    def test_stride_fractions_valid(self):
        """Stride fractions in (0, 1)."""
        for speed, frac in WINDOW_CONFIG['stride_fraction'].items():
            assert 0 < frac < 1
    
    def test_bounds_sensible(self):
        """Min < max."""
        assert WINDOW_CONFIG['min_window'] < WINDOW_CONFIG['max_window']
