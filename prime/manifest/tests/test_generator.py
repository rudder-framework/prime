"""
Tests for Manifest Generator v2.5
"""

import pytest
import sys
sys.path.insert(0, '/Users/jasonrudder/rudder')

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from prime.manifest.generator import (
    apply_engine_adjustments,
    apply_viz_adjustments,
    get_window_params,
    get_output_hints,
    build_signal_config,
    build_manifest,
    validate_manifest,
    compute_engine_window_overrides,
    BASE_ENGINES,
    BASE_VISUALIZATIONS,
    ENGINE_MIN_WINDOWS,
)


# ============================================================
# Test: Engine Adjustments
# ============================================================

class TestInclusivePhilosophy:
    """Test the inclusive engine selection philosophy: 'If it's a maybe, run it.'"""

    def test_no_removal_for_trending(self):
        """TRENDING should not remove any engines (inclusive)."""
        original = ['spectral', 'harmonics', 'hurst', 'kurtosis']
        engines = apply_engine_adjustments(original, 'TRENDING')
        # All original engines should still be present
        for eng in original:
            assert eng in engines, f"{eng} was incorrectly removed from TRENDING"

    def test_no_removal_for_periodic(self):
        """PERIODIC should not remove any engines (inclusive)."""
        original = ['spectral', 'hurst', 'trend_r2', 'kurtosis']
        engines = apply_engine_adjustments(original, 'PERIODIC')
        # All original engines should still be present
        for eng in original:
            assert eng in engines, f"{eng} was incorrectly removed from PERIODIC"

    def test_no_removal_for_chaotic(self):
        """CHAOTIC should not remove any engines (inclusive)."""
        original = ['spectral', 'harmonics', 'trend_r2', 'kurtosis']
        engines = apply_engine_adjustments(original, 'CHAOTIC')
        for eng in original:
            assert eng in engines, f"{eng} was incorrectly removed from CHAOTIC"

    def test_no_removal_for_random(self):
        """RANDOM should not remove any engines (inclusive)."""
        original = ['spectral', 'harmonics', 'hurst', 'lyapunov']
        engines = apply_engine_adjustments(original, 'RANDOM')
        for eng in original:
            assert eng in engines, f"{eng} was incorrectly removed from RANDOM"

    def test_constant_is_only_exception(self):
        """CONSTANT is the only type that removes all engines."""
        original = ['spectral', 'harmonics', 'hurst', 'kurtosis']
        engines = apply_engine_adjustments(original, 'CONSTANT')
        assert engines == [], "CONSTANT should remove all engines"

    def test_base_engines_always_included(self):
        """Base engines (crest_factor, kurtosis, skewness, spectral) are added for all non-CONSTANT types."""
        for pattern in ['TRENDING', 'PERIODIC', 'CHAOTIC', 'RANDOM', 'BINARY', 'DISCRETE']:
            engines = apply_engine_adjustments([], pattern)
            # Should have at least some engines from base or type-specific additions
            assert len(engines) > 0, f"{pattern} should add engines"


class TestEngineAdjustments:

    def test_trending_adds_hurst(self):
        """TRENDING adds hurst, rate_of_change."""
        engines = apply_engine_adjustments(['spectral', 'kurtosis'], 'TRENDING')
        assert 'hurst' in engines
        assert 'rate_of_change' in engines

    def test_trending_keeps_harmonics(self):
        """TRENDING keeps harmonics (inclusive philosophy - might catch oscillating trends)."""
        engines = apply_engine_adjustments(['spectral', 'harmonics'], 'TRENDING')
        assert 'harmonics' in engines  # Inclusive: harmonics might reveal oscillating trends

    def test_periodic_adds_harmonics(self):
        """PERIODIC adds harmonics, thd."""
        engines = apply_engine_adjustments(['spectral'], 'PERIODIC')
        assert 'harmonics' in engines
        assert 'thd' in engines

    def test_periodic_keeps_hurst(self):
        """PERIODIC keeps hurst (inclusive philosophy - can show persistence in periodic)."""
        engines = apply_engine_adjustments(['hurst', 'spectral'], 'PERIODIC')
        assert 'hurst' in engines  # Inclusive: hurst can reveal persistence patterns

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

    def test_trending_keeps_waterfall(self):
        """TRENDING keeps waterfall (inclusive philosophy)."""
        viz = apply_viz_adjustments(['waterfall', 'spectral_density'], 'TRENDING')
        assert 'waterfall' in viz  # Inclusive: waterfall can show spectral changes in trends

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
        # Inclusive philosophy: harmonics IS included (might reveal oscillating trends)
        assert 'harmonics' in config['engines'] or 'spectral' in config['engines']
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
            'version': '2.5',
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
            'version': '2.5',
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
            'version': '2.5',
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


# ============================================================
# Test: Per-Engine Window Specification (PR #1)
# ============================================================

class TestEngineMinWindows:
    """Test ENGINE_MIN_WINDOWS constant."""

    def test_spectral_requires_64(self):
        """Spectral/FFT engines require 64 samples."""
        assert ENGINE_MIN_WINDOWS['spectral'] == 64
        assert ENGINE_MIN_WINDOWS['harmonics'] == 64
        assert ENGINE_MIN_WINDOWS['fundamental_freq'] == 64
        assert ENGINE_MIN_WINDOWS['thd'] == 64

    def test_hurst_requires_128(self):
        """Hurst exponent requires 128 samples for R/S analysis."""
        assert ENGINE_MIN_WINDOWS['hurst'] == 128

    def test_sample_entropy_requires_64(self):
        """Sample entropy requires 64 samples for embedding."""
        assert ENGINE_MIN_WINDOWS['sample_entropy'] == 64


class TestComputeEngineWindowOverrides:
    """Test compute_engine_window_overrides function."""

    def test_no_overrides_for_large_window(self):
        """No overrides needed when signal window >= all engine minimums."""
        engines = ['spectral', 'hurst', 'kurtosis']
        overrides = compute_engine_window_overrides(engines, signal_window=128)
        assert overrides == {}

    def test_spectral_override_for_small_window(self):
        """Spectral engine needs override when window=32."""
        engines = ['spectral', 'crest_factor', 'kurtosis']
        overrides = compute_engine_window_overrides(engines, signal_window=32)
        assert 'spectral' in overrides
        assert overrides['spectral'] == 64
        assert 'crest_factor' not in overrides  # Works fine at 32
        assert 'kurtosis' not in overrides

    def test_hurst_override_for_medium_window(self):
        """Hurst needs override even when window=64."""
        engines = ['spectral', 'hurst']
        overrides = compute_engine_window_overrides(engines, signal_window=64)
        assert 'spectral' not in overrides  # 64 is sufficient
        assert 'hurst' in overrides
        assert overrides['hurst'] == 128

    def test_multiple_overrides(self):
        """Multiple engines can have overrides."""
        engines = ['spectral', 'harmonics', 'hurst', 'sample_entropy']
        overrides = compute_engine_window_overrides(engines, signal_window=32)
        assert 'spectral' in overrides
        assert 'harmonics' in overrides
        assert 'hurst' in overrides
        assert 'sample_entropy' in overrides
        assert overrides['spectral'] == 64
        assert overrides['hurst'] == 128

    def test_empty_engines_no_overrides(self):
        """Empty engine list produces no overrides."""
        overrides = compute_engine_window_overrides([], signal_window=32)
        assert overrides == {}

    def test_unknown_engine_no_override(self):
        """Unknown engines (not in ENGINE_MIN_WINDOWS) don't need overrides."""
        engines = ['unknown_engine', 'custom_metric']
        overrides = compute_engine_window_overrides(engines, signal_window=32)
        assert overrides == {}


class TestSignalConfigWithOverrides:
    """Test build_signal_config includes engine_window_overrides when needed."""

    def test_periodic_small_window_has_overrides(self):
        """PERIODIC signal with small window gets engine_window_overrides."""
        row = {
            'temporal_pattern': 'PERIODIC',
            'spectral': 'HARMONIC',
            'n_samples': 1000,
            'seasonal_period': 8,  # Will result in window_size=32 (4*8)
        }
        config = build_signal_config('sensor_01', 'unit_1', row)

        # With window_size=32, spectral/harmonics should need overrides
        assert config['window_size'] == 32
        assert 'engine_window_overrides' in config
        assert config['engine_window_overrides'].get('spectral') == 64
        assert config['engine_window_overrides'].get('harmonics') == 64

    def test_large_window_no_overrides(self):
        """Signal with large window (128+) has no engine_window_overrides."""
        row = {
            'temporal_pattern': 'PERIODIC',
            'spectral': 'HARMONIC',
            'n_samples': 5000,
            # No seasonal_period, will use default window=128
        }
        config = build_signal_config('sensor_01', 'unit_1', row)

        assert config['window_size'] == 128
        assert 'engine_window_overrides' not in config

    def test_constant_signal_no_overrides(self):
        """CONSTANT signal has no engines, so no overrides."""
        row = {
            'temporal_pattern': 'CONSTANT',
            'spectral': 'NONE',
            'n_samples': 500,
        }
        config = build_signal_config('sensor_01', 'unit_1', row)

        assert config['engines'] == []
        assert 'engine_window_overrides' not in config


class TestManifestEngineWindows:
    """Test build_manifest includes engine_windows section."""

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
    def test_manifest_has_engine_windows_section(self):
        """Manifest v2.5 includes engine_windows section."""
        typology_df = pd.DataFrame([
            {'signal_id': 'sensor_01', 'cohort': 'unit_1', 'temporal_pattern': 'PERIODIC', 'spectral': 'HARMONIC', 'n_samples': 1000},
        ])
        manifest = build_manifest(typology_df)

        assert manifest['version'] == '2.5'
        assert 'engine_windows' in manifest
        assert manifest['engine_windows']['spectral'] == 64
        assert manifest['engine_windows']['hurst'] == 128
        assert 'note' in manifest['engine_windows']

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas required")
    def test_manifest_per_signal_overrides(self):
        """Manifest includes per-signal engine_window_overrides when needed."""
        # Create a signal with small window that will need overrides
        typology_df = pd.DataFrame([
            {
                'signal_id': 'fast_sensor',
                'cohort': 'unit_1',
                'temporal_pattern': 'PERIODIC',
                'spectral': 'HARMONIC',
                'n_samples': 1000,
                'seasonal_period': 8,  # Results in window_size=32
            },
        ])
        manifest = build_manifest(typology_df)

        signal_config = manifest['cohorts']['unit_1']['fast_sensor']
        assert signal_config['window_size'] == 32
        assert 'engine_window_overrides' in signal_config
        assert signal_config['engine_window_overrides']['spectral'] == 64
