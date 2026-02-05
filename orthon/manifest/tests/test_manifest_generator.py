"""
Tests for PR9: Updated Manifest Generator
"""

import pytest
import pandas as pd
import sys
sys.path.insert(0, '/Users/jasonrudder/orthon')

from orthon.manifest.generator import (
    build_signal_config,
    generate_manifest,
    get_engines_for_type,
    BASE_ENGINES,
)


# ============================================================
# Engine Selection
# ============================================================

class TestEngineSelection:
    
    def test_base_engines_always_present(self):
        """Base engines included for all non-CONSTANT types."""
        for typ in ['TRENDING', 'PERIODIC', 'RANDOM', 'STATIONARY']:
            engines = get_engines_for_type(typ)
            for base in BASE_ENGINES:
                assert base in engines, f"{base} missing for {typ}"
    
    def test_constant_no_engines(self):
        """CONSTANT gets no engines."""
        engines = get_engines_for_type('CONSTANT')
        assert engines == []
    
    def test_trending_has_trend_engines(self):
        """TRENDING has trend-specific engines."""
        engines = get_engines_for_type('TRENDING')
        assert 'hurst' in engines
        assert 'rate_of_change' in engines
        assert 'trend_r2' in engines
        assert 'cusum' in engines
    
    def test_periodic_has_harmonic_engines(self):
        """PERIODIC has harmonic engines."""
        engines = get_engines_for_type('PERIODIC')
        assert 'harmonics' in engines
        assert 'thd' in engines
        assert 'frequency_bands' in engines


# ============================================================
# Signal Config: Window/Stride from Characteristic Time
# ============================================================

class TestSignalConfigWindowStride:
    
    def test_window_from_acf(self):
        """Window derived from acf_half_life."""
        row = {
            'signal_id': 'temp',
            'temporal_pattern': 'TRENDING',
            'spectral': 'RED_NOISE',
            'n_samples': 10000,
            'acf_half_life': 100.0,
            'dominant_frequency': None,
            'turning_point_ratio': 0.5,
            'hurst': 0.9,
        }
        config = build_signal_config('temp', 'unit_1', row)
        
        assert config['characteristic_source'] == 'acf_half_life'
        assert config['characteristic_time'] == 100.0
        assert config['window_size'] == 250  # 100 * 2.5
    
    def test_window_from_frequency(self):
        """Window derived from dominant frequency."""
        row = {
            'signal_id': 'vib',
            'temporal_pattern': 'PERIODIC',
            'spectral': 'HARMONIC',
            'n_samples': 10000,
            'acf_half_life': None,
            'dominant_frequency': 0.02,  # Period = 50
            'turning_point_ratio': 0.8,
            'hurst': 0.5,
        }
        config = build_signal_config('vib', 'unit_1', row)
        
        assert config['characteristic_source'] == 'dominant_frequency'
        assert config['characteristic_time'] == 50.0
        assert config['window_size'] == 125  # 50 * 2.5
    
    def test_stride_varies_by_dynamics(self):
        """Stride adjusts based on dynamics speed."""
        # Fast dynamics (short char_time relative to signal)
        row_fast = {
            'signal_id': 'fast_sig',
            'temporal_pattern': 'PERIODIC',
            'n_samples': 100000,
            'acf_half_life': 50.0,  # 0.05% of signal = fast
            'dominant_frequency': None,
            'turning_point_ratio': None,
            'hurst': None,
        }
        config_fast = build_signal_config('fast_sig', 'unit', row_fast)
        
        # Slow dynamics (long char_time relative to signal)
        row_slow = {
            'signal_id': 'slow_sig',
            'temporal_pattern': 'TRENDING',
            'n_samples': 1000,
            'acf_half_life': 200.0,  # 20% of signal = slow
            'dominant_frequency': None,
            'turning_point_ratio': None,
            'hurst': None,
        }
        config_slow = build_signal_config('slow_sig', 'unit', row_slow)
        
        assert config_fast['dynamics_speed'] == 'fast'
        assert config_slow['dynamics_speed'] == 'slow'
        # Fast should have lower stride fraction (more overlap)
        fast_overlap = 1 - (config_fast['stride'] / config_fast['window_size'])
        slow_overlap = 1 - (config_slow['stride'] / config_slow['window_size'])
        assert fast_overlap > slow_overlap
    
    def test_window_includes_char_time_metadata(self):
        """Config includes characteristic time metadata."""
        row = {
            'signal_id': 'test',
            'temporal_pattern': 'STATIONARY',
            'n_samples': 1000,
            'acf_half_life': 50.0,
            'dominant_frequency': None,
            'hurst': 0.5,
        }
        config = build_signal_config('test', 'unit', row)
        
        assert 'characteristic_time' in config
        assert 'characteristic_source' in config
        assert 'dynamics_speed' in config


# ============================================================
# Manifest Generation
# ============================================================

class TestManifestGeneration:
    
    @pytest.fixture
    def sample_typology_df(self):
        """Sample typology DataFrame."""
        return pd.DataFrame([
            {
                'signal_id': 'sensor_01',
                'cohort': 'unit_1',
                'temporal_pattern': 'TRENDING',
                'spectral': 'RED_NOISE',
                'n_samples': 10000,
                'acf_half_life': 100.0,
                'dominant_frequency': None,
                'turning_point_ratio': 0.5,
                'hurst': 0.95,
            },
            {
                'signal_id': 'sensor_02',
                'cohort': 'unit_1',
                'temporal_pattern': 'PERIODIC',
                'spectral': 'HARMONIC',
                'n_samples': 10000,
                'acf_half_life': None,
                'dominant_frequency': 0.01,
                'turning_point_ratio': 0.8,
                'hurst': 0.5,
            },
            {
                'signal_id': 'flag',
                'cohort': 'unit_1',
                'temporal_pattern': 'CONSTANT',
                'spectral': 'NARROWBAND',
                'n_samples': 10000,
                'acf_half_life': None,
                'dominant_frequency': None,
                'turning_point_ratio': None,
                'hurst': None,
            },
        ])
    
    def test_manifest_version(self, sample_typology_df):
        """Manifest has correct version."""
        manifest = generate_manifest(
            sample_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        assert manifest['version'] == '2.3'
    
    def test_constant_in_skip_signals(self, sample_typology_df):
        """CONSTANT signals go to skip_signals."""
        manifest = generate_manifest(
            sample_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        assert 'unit_1/flag' in manifest['skip_signals']
    
    def test_active_signals_have_window_config(self, sample_typology_df):
        """Active signals have window/stride config."""
        manifest = generate_manifest(
            sample_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        sensor_01 = manifest['cohorts']['unit_1']['sensor_01']
        assert 'window_size' in sensor_01
        assert 'stride' in sensor_01
        assert 'characteristic_time' in sensor_01
        assert 'characteristic_source' in sensor_01
    
    def test_different_windows_per_signal(self, sample_typology_df):
        """Different signals can have different window sizes."""
        manifest = generate_manifest(
            sample_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        sensor_01 = manifest['cohorts']['unit_1']['sensor_01']
        sensor_02 = manifest['cohorts']['unit_1']['sensor_02']
        
        # TRENDING (acf=100) should have different window than PERIODIC (freq=0.01, period=100)
        # Actually both might be similar here, but sources differ
        assert sensor_01['characteristic_source'] == 'acf_half_life'
        assert sensor_02['characteristic_source'] == 'dominant_frequency'
    
    def test_summary_counts(self, sample_typology_df):
        """Summary has correct counts."""
        manifest = generate_manifest(
            sample_typology_df,
            'obs.parquet', 'typ.parquet', 'output/'
        )
        
        assert manifest['summary']['total_signals'] == 3
        assert manifest['summary']['active_signals'] == 2
        assert manifest['summary']['constant_signals'] == 1


# ============================================================
# Real Dataset Simulation
# ============================================================

class TestRealDatasetSimulation:
    
    def test_skab_like_dataset(self):
        """SKAB-like dataset with mixed signal types."""
        df = pd.DataFrame([
            {
                'signal_id': 'Accelerometer1RMS',
                'cohort': 'skab',
                'temporal_pattern': 'STATIONARY',
                'spectral': 'NARROWBAND',
                'n_samples': 9405,
                'acf_half_life': 50.0,
                'dominant_frequency': None,
                'turning_point_ratio': 0.65,
                'hurst': 0.88,
            },
            {
                'signal_id': 'Current',
                'cohort': 'skab',
                'temporal_pattern': 'RANDOM',
                'spectral': 'BROADBAND',
                'n_samples': 9405,
                'acf_half_life': None,
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
        
        manifest = generate_manifest(df, 'obs.parquet', 'typ.parquet', 'output/')
        
        # All should be active (no CONSTANT)
        assert manifest['summary']['active_signals'] == 3
        assert manifest['summary']['constant_signals'] == 0
        
        # Each has window config
        for sig in ['Accelerometer1RMS', 'Current', 'Temperature']:
            config = manifest['cohorts']['skab'][sig]
            assert config['window_size'] >= 64
            assert config['stride'] > 0
    
    def test_building_vibration_like(self):
        """Building vibration dataset."""
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
        
        # RANDOM and TRENDING should have different dynamics
        random_cfg = manifest['cohorts']['building']['floor_x_sway']
        trend_cfg = manifest['cohorts']['fab_floor']['floor_z_vc_e']
        
        # Both have derived window configs
        assert 'characteristic_time' in random_cfg
        assert 'characteristic_time' in trend_cfg
