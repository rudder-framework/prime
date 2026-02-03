"""
Tests for PR10: System Window + Representation Type
"""

import pytest
import math
import pandas as pd
import sys
sys.path.insert(0, '/Users/jasonrudder/orthon')

from orthon.window.system_window import (
    compute_system_window,
    classify_representation,
    compute_signal_representation,
    compute_system_representation,
    summarize_representations,
    REPRESENTATION_CONFIG,
)

from orthon.window.manifest_generator import (
    generate_manifest,
    get_engines_for_type,
)


# ============================================================
# System Window Computation
# ============================================================

class TestSystemWindow:
    
    def test_max_method(self):
        """System window = max characteristic time."""
        char_times = [10, 50, 100, 200]
        system_window = compute_system_window(char_times, method='max')
        # 200 * 2.5 = 500
        assert system_window == 500
    
    def test_empty_list(self):
        """Empty list returns minimum."""
        system_window = compute_system_window([])
        assert system_window == REPRESENTATION_CONFIG['min_system_window']
    
    def test_invalid_values_filtered(self):
        """NaN and None values filtered."""
        char_times = [None, math.nan, 100, 0, -10]
        system_window = compute_system_window(char_times)
        # Only 100 is valid, 100 * 2.5 = 250
        assert system_window == 250
    
    def test_bounds_applied(self):
        """System window bounded by min/max."""
        # Very small
        small = compute_system_window([1])
        assert small >= REPRESENTATION_CONFIG['min_system_window']
        
        # Very large
        large = compute_system_window([10000])
        assert large <= REPRESENTATION_CONFIG['max_system_window']
    
    def test_median_method(self):
        """Median method works."""
        char_times = [10, 100, 1000]  # Median = 100
        system_window = compute_system_window(char_times, method='median')
        assert system_window == 250  # 100 * 2.5


# ============================================================
# Representation Classification
# ============================================================

class TestRepresentationClassification:
    
    def test_fast_signal_spectral(self):
        """Fast signal (low τ/W ratio) → spectral."""
        rep = classify_representation(
            characteristic_time=10,
            system_window=1000,
        )
        # 10/1000 = 0.01 < 0.3 → spectral
        assert rep == 'spectral'
    
    def test_slow_signal_trajectory(self):
        """Slow signal (high τ/W ratio) → trajectory."""
        rep = classify_representation(
            characteristic_time=500,
            system_window=1000,
        )
        # 500/1000 = 0.5 > 0.3 → trajectory
        assert rep == 'trajectory'
    
    def test_boundary_case(self):
        """At threshold boundary."""
        threshold = REPRESENTATION_CONFIG['spectral_threshold']
        
        # Just below threshold → spectral
        rep_below = classify_representation(
            characteristic_time=threshold * 1000 - 1,
            system_window=1000,
        )
        assert rep_below == 'spectral'
        
        # At threshold → trajectory
        rep_at = classify_representation(
            characteristic_time=threshold * 1000,
            system_window=1000,
        )
        assert rep_at == 'trajectory'
    
    def test_nan_defaults_spectral(self):
        """NaN characteristic time defaults to spectral."""
        rep = classify_representation(
            characteristic_time=math.nan,
            system_window=1000,
        )
        assert rep == 'spectral'


# ============================================================
# Signal Representation Config
# ============================================================

class TestSignalRepresentation:
    
    def test_spectral_has_bands(self):
        """Spectral representation includes bands."""
        cfg = compute_signal_representation(
            characteristic_time=10,
            system_window=1000,
        )
        assert cfg['representation'] == 'spectral'
        assert 'bands' in cfg
        assert len(cfg['bands']) > 0
    
    def test_trajectory_features(self):
        """Trajectory representation has trajectory features."""
        cfg = compute_signal_representation(
            characteristic_time=500,
            system_window=1000,
        )
        assert cfg['representation'] == 'trajectory'
        assert 'value_start' in cfg['features'] or 'slope' in cfg['features']
    
    def test_tau_ratio_included(self):
        """τ/W ratio included in config."""
        cfg = compute_signal_representation(
            characteristic_time=100,
            system_window=1000,
        )
        assert 'tau_ratio' in cfg
        assert abs(cfg['tau_ratio'] - 0.1) < 0.001


# ============================================================
# Batch Processing
# ============================================================

class TestBatchProcessing:
    
    def test_skab_like_signals(self):
        """Process SKAB-like signal set."""
        signals = [
            {'signal_id': 'Current', 'characteristic_time': 1.0},
            {'signal_id': 'Voltage', 'characteristic_time': 1.0},
            {'signal_id': 'Temperature', 'characteristic_time': 941.0},
            {'signal_id': 'Thermocouple', 'characteristic_time': 2419.0},
        ]
        
        result = compute_system_representation(signals)
        
        # System window from max (2419 * 2.5 = 6047, capped at 4096)
        assert result['system_window'] == 4096
        
        # Fast signals → spectral (τ/W << 0.3)
        assert result['signals']['Current']['representation'] == 'spectral'
        assert result['signals']['Voltage']['representation'] == 'spectral'
        
        # Temperature: 941/4096 = 0.23 < 0.3 → spectral
        assert result['signals']['Temperature']['representation'] == 'spectral'
        
        # Thermocouple: 2419/4096 = 0.59 > 0.3 → trajectory
        assert result['signals']['Thermocouple']['representation'] == 'trajectory'
    
    def test_summary(self):
        """Summary counts correct."""
        signals = [
            {'signal_id': 'fast1', 'characteristic_time': 10},
            {'signal_id': 'fast2', 'characteristic_time': 20},
            {'signal_id': 'slow1', 'characteristic_time': 500},
        ]
        
        result = compute_system_representation(signals)
        summary = summarize_representations(result)
        
        assert summary['n_spectral'] == 2
        assert summary['n_trajectory'] == 1
        assert 'fast1' in summary['spectral_signals']
        assert 'slow1' in summary['trajectory_signals']


# ============================================================
# Manifest Generation
# ============================================================

class TestManifestGeneration:
    
    @pytest.fixture
    def skab_typology_df(self):
        """SKAB-like typology DataFrame."""
        return pd.DataFrame([
            {
                'signal_id': 'Current',
                'cohort': 'skab',
                'temporal_pattern': 'RANDOM',
                'spectral': 'BROADBAND',
                'n_samples': 9405,
                'acf_half_life': 1.0,
                'dominant_frequency': None,
                'turning_point_ratio': 0.67,
                'hurst': 0.5,
            },
            {
                'signal_id': 'Voltage',
                'cohort': 'skab',
                'temporal_pattern': 'RANDOM',
                'spectral': 'BROADBAND',
                'n_samples': 9405,
                'acf_half_life': 1.0,
                'dominant_frequency': None,
                'turning_point_ratio': 0.67,
                'hurst': 0.5,
            },
            {
                'signal_id': 'Temperature',
                'cohort': 'skab',
                'temporal_pattern': 'TRENDING',
                'spectral': 'RED_NOISE',
                'n_samples': 9405,
                'acf_half_life': None,
                'dominant_frequency': None,
                'turning_point_ratio': 0.75,
                'hurst': 1.0,
            },
        ])
    
    def test_manifest_version(self, skab_typology_df):
        """Manifest version is 2.4."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        assert manifest['version'] == '2.4'
    
    def test_system_window_in_manifest(self, skab_typology_df):
        """System window present in manifest."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        assert 'system' in manifest
        assert 'window' in manifest['system']
        assert manifest['system']['window'] >= 64
    
    def test_representation_in_signal_config(self, skab_typology_df):
        """Each signal has representation config."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        current = manifest['cohorts']['skab']['Current']
        temp = manifest['cohorts']['skab']['Temperature']
        
        assert 'representation' in current
        assert 'representation' in temp
        assert 'state_features' in current
        assert 'state_features' in temp
    
    def test_fast_slow_representations(self, skab_typology_df):
        """Fast signals spectral, slow signals trajectory."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        current = manifest['cohorts']['skab']['Current']
        temp = manifest['cohorts']['skab']['Temperature']
        
        assert current['representation'] == 'spectral'
        assert temp['representation'] == 'trajectory'
    
    def test_spectral_has_bands(self, skab_typology_df):
        """Spectral signals have bands config."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        current = manifest['cohorts']['skab']['Current']
        assert 'bands' in current
    
    def test_representation_summary(self, skab_typology_df):
        """Representation summary in manifest."""
        manifest = generate_manifest(
            skab_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        assert 'representation_summary' in manifest
        summary = manifest['representation_summary']
        assert 'n_spectral' in summary
        assert 'n_trajectory' in summary
        assert 'spectral_signals' in summary
        assert 'trajectory_signals' in summary


# ============================================================
# Integration: Full Pipeline
# ============================================================

class TestFullPipeline:
    
    def test_building_vibration_like(self):
        """Building vibration dataset simulation."""
        df = pd.DataFrame([
            {
                'signal_id': 'floor_x_sway',
                'cohort': 'building',
                'temporal_pattern': 'RANDOM',
                'spectral': 'BROADBAND',
                'n_samples': 10000,
                'acf_half_life': None,
                'dominant_frequency': None,
                'turning_point_ratio': 0.67,
                'hurst': 0.5,
            },
            {
                'signal_id': 'floor_z_vc_e',
                'cohort': 'fab_floor',
                'temporal_pattern': 'TRENDING',
                'spectral': 'RED_NOISE',
                'n_samples': 10000,
                'acf_half_life': None,
                'dominant_frequency': None,
                'turning_point_ratio': 0.75,
                'hurst': 1.0,
            },
        ])
        
        manifest = generate_manifest(df, 'obs.parquet', 'typ.parquet', 'output/')
        
        # Should have system window
        assert manifest['system']['window'] >= 64
        
        # Both signals should have representations
        sway = manifest['cohorts']['building']['floor_x_sway']
        vc = manifest['cohorts']['fab_floor']['floor_z_vc_e']
        
        assert 'representation' in sway
        assert 'representation' in vc


# ============================================================
# Config Validation
# ============================================================

class TestConfig:
    
    def test_threshold_valid(self):
        """Spectral threshold in (0, 1)."""
        threshold = REPRESENTATION_CONFIG['spectral_threshold']
        assert 0 < threshold < 1
    
    def test_bands_defined(self):
        """Default bands defined."""
        bands = REPRESENTATION_CONFIG['default_bands']
        assert len(bands) > 0
        assert all(0 < b <= 0.5 for b in bands)
    
    def test_features_defined(self):
        """Spectral and trajectory features defined."""
        assert len(REPRESENTATION_CONFIG['spectral_features']) > 0
        assert len(REPRESENTATION_CONFIG['trajectory_features']) > 0
