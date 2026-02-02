"""
Tests for Manifest Generator v2.2
"""

import pytest
import sys
sys.path.insert(0, '/Users/jasonrudder/orthon')

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from orthon.manifest.generator import (
    apply_engine_adjustments,
    apply_viz_adjustments,
    get_window_params,
    get_output_hints,
    build_signal_config,
    build_manifest,
    validate_manifest,
    BASE_ENGINES,
    BASE_VISUALIZATIONS,
)


# ============================================================
# Test: Engine Adjustments
# ============================================================

class TestEngineAdjustments:

    def test_trending_adds_hurst(self):
        """TRENDING adds hurst, rate_of_change."""
        engines = apply_engine_adjustments(['spectral', 'kurtosis'], 'TRENDING')
        assert 'hurst' in engines
        assert 'rate_of_change' in engines

    def test_trending_removes_harmonics(self):
        """TRENDING removes harmonics."""
        engines = apply_engine_adjustments(['spectral', 'harmonics'], 'TRENDING')
        assert 'harmonics' not in engines

    def test_periodic_adds_harmonics(self):
        """PERIODIC adds harmonics, thd."""
        engines = apply_engine_adjustments(['spectral'], 'PERIODIC')
        assert 'harmonics' in engines
        assert 'thd' in engines

    def test_periodic_removes_hurst(self):
        """PERIODIC removes hurst."""
        engines = apply_engine_adjustments(['hurst', 'spectral'], 'PERIODIC')
        assert 'hurst' not in engines

    def test_constant_removes_all(self):
        """CONSTANT removes all engines."""
        engines = apply_engine_adjustments(['hurst', 'spectral', 'kurtosis'], 'CONSTANT')
        assert engines == []

    def test_binary_adds_transition_count(self):
        """BINARY adds transition analysis engines."""
        engines = apply_engine_adjustments(['kurtosis'], 'BINARY')
        assert 'transition_count' in engines
        assert 'duty_cycle' in engines

    def test_event_adds_event_engines(self):
        """EVENT adds event analysis engines."""
        engines = apply_engine_adjustments(['kurtosis'], 'EVENT')
        assert 'event_rate' in engines
        assert 'inter_event_time' in engines

    def test_unknown_type_no_change(self):
        """Unknown type leaves engines unchanged."""
        original = ['a', 'b', 'c']
        engines = apply_engine_adjustments(original, 'UNKNOWN_TYPE')
        assert engines == original


# ============================================================
# Test: Visualization Adjustments
# ============================================================

class TestVizAdjustments:

    def test_trending_adds_trend_overlay(self):
        """TRENDING adds trend_overlay."""
        viz = apply_viz_adjustments(['spectral_density'], 'TRENDING')
        assert 'trend_overlay' in viz

    def test_trending_removes_waterfall(self):
        """TRENDING removes waterfall."""
        viz = apply_viz_adjustments(['waterfall', 'spectral_density'], 'TRENDING')
        assert 'waterfall' not in viz

    def test_periodic_adds_waterfall(self):
        """PERIODIC adds waterfall."""
        viz = apply_viz_adjustments(['spectral_density'], 'PERIODIC')
        assert 'waterfall' in viz

    def test_constant_removes_all(self):
        """CONSTANT removes all visualizations."""
        viz = apply_viz_adjustments(['waterfall', 'spectral_density'], 'CONSTANT')
        assert viz == []

    def test_binary_adds_state_timeline(self):
        """BINARY adds state_timeline."""
        viz = apply_viz_adjustments([], 'BINARY')
        assert 'state_timeline' in viz


# ============================================================
# Test: Window Parameters
# ============================================================

class TestWindowParams:

    def test_trending_small_stride(self):
        """TRENDING gets smaller stride (more overlap)."""
        params = get_window_params('TRENDING', 1000)
        assert params['stride'] == 32
        assert params['derivative_depth'] == 2

    def test_constant_full_window(self):
        """CONSTANT gets full-signal window."""
        params = get_window_params('CONSTANT', 500)
        assert params['window_size'] == 500
        assert params['stride'] == 500

    def test_periodic_standard(self):
        """PERIODIC gets standard windowing."""
        params = get_window_params('PERIODIC', 1000)
        assert params['window_size'] == 128
        assert params['stride'] == 64

    def test_impulsive_small_window(self):
        """IMPULSIVE gets small windows."""
        params = get_window_params('IMPULSIVE', 1000)
        assert params['window_size'] == 64
        assert params['stride'] == 16


# ============================================================
# Test: Output Hints
# ============================================================

class TestOutputHints:

    def test_periodic_per_bin(self):
        """PERIODIC gets per_bin spectral output."""
        hints = get_output_hints('PERIODIC', 'NARROWBAND')
        assert hints['spectral']['output_mode'] == 'per_bin'

    def test_trending_summary(self):
        """TRENDING gets summary spectral output."""
        hints = get_output_hints('TRENDING', 'RED_NOISE')
        assert hints['spectral']['output_mode'] == 'summary'

    def test_harmonic_adds_harmonics_config(self):
        """HARMONIC spectral adds harmonics config."""
        hints = get_output_hints('PERIODIC', 'HARMONIC')
        assert 'harmonics' in hints
        assert hints['harmonics']['n_harmonics'] == 5


# ============================================================
# Test: Build Signal Config
# ============================================================

class TestBuildSignalConfig:

    def test_trending_signal(self):
        """Build config for TRENDING signal."""
        row = {
            'temporal_pattern': 'TRENDING',
            'spectral': 'RED_NOISE',
            'n_samples': 1000,
        }
        config = build_signal_config('sensor_01', 'unit_1', row)

        assert config['typology']['temporal_pattern'] == 'TRENDING'
        assert 'hurst' in config['engines']
        assert 'harmonics' not in config['engines']
        assert config['stride'] == 32
        assert 'trend_overlay' in config['visualizations']

    def test_periodic_signal(self):
        """Build config for PERIODIC signal."""
        row = {
            'temporal_pattern': 'PERIODIC',
            'spectral': 'HARMONIC',
            'n_samples': 5000,
        }
        config = build_signal_config('acc_x', 'bearing_1', row)

        assert 'harmonics' in config['engines']
        assert 'waterfall' in config['visualizations']
        assert 'harmonics' in config['output_hints']

    def test_constant_signal(self):
        """Build config for CONSTANT signal."""
        row = {
            'temporal_pattern': 'CONSTANT',
            'spectral': 'NONE',
            'n_samples': 500,
        }
        config = build_signal_config('Mn_II', 'CL10', row)

        assert config['engines'] == []
        assert config['visualizations'] == []
        assert config['is_discrete_sparse'] is True

    def test_binary_signal(self):
        """Build config for BINARY signal."""
        row = {
            'temporal_pattern': 'BINARY',
            'spectral': 'SWITCHING',
            'n_samples': 10000,
        }
        config = build_signal_config('switch', 'room_1', row)

        assert 'transition_count' in config['engines']
        assert config['is_discrete_sparse'] is True


# ============================================================
# Test: Validate Manifest
# ============================================================

class TestValidateManifest:

    def test_valid_manifest(self):
        """Valid manifest passes validation."""
        manifest = {
            'version': '2.2',
            'job_id': 'test-123',
            'paths': {'observations': 'obs.parquet'},
            'summary': {'total_signals': 1},
            'cohorts': {
                'unit_1': {
                    'sensor_01': {
                        'engines': ['spectral'],
                        'typology': {'temporal_pattern': 'TRENDING'},
                    }
                }
            },
            'skip_signals': [],
        }
        errors = validate_manifest(manifest)
        assert errors == []

    def test_missing_required_key(self):
        """Missing required key fails validation."""
        manifest = {
            'version': '2.2',
            # missing job_id, paths, summary, cohorts
        }
        errors = validate_manifest(manifest)
        assert len(errors) > 0
        assert any('job_id' in e for e in errors)

    def test_missing_engines(self):
        """Missing engines in signal fails validation."""
        manifest = {
            'version': '2.2',
            'job_id': 'test',
            'paths': {},
            'summary': {},
            'cohorts': {
                'unit_1': {
                    'sensor_01': {
                        'typology': {'temporal_pattern': 'TRENDING'},
                        # missing engines
                    }
                }
            },
        }
        errors = validate_manifest(manifest)
        assert any('engines' in e for e in errors)

    def test_constant_in_cohorts_fails(self):
        """CONSTANT signal in cohorts (not skip_signals) fails."""
        manifest = {
            'version': '2.2',
            'job_id': 'test',
            'paths': {},
            'summary': {},
            'cohorts': {
                'unit_1': {
                    'sensor_01': {
                        'engines': [],
                        'typology': {'temporal_pattern': 'CONSTANT'},
                    }
                }
            },
            'skip_signals': [],
        }
        errors = validate_manifest(manifest)
        assert any('CONSTANT' in e for e in errors)
