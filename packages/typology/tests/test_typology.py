"""Tests for typology package."""

import numpy as np
import typology


class TestWindowFromLength:

    def test_short_signal(self):
        ws = typology.window_from_length(50)
        assert ws['source'] == 'too_short'
        assert ws['window_size'] == 50

    def test_medium_signal(self):
        ws = typology.window_from_length(300)
        assert ws['window_size'] == 64
        assert ws['stride'] == 16

    def test_long_signal(self):
        ws = typology.window_from_length(5000)
        assert ws['window_size'] == 256
        assert ws['stride'] == 64


class TestFromObservations:

    def test_constant_signal(self):
        values = np.ones(1000) * 42.0
        result = typology.from_observations(values)
        assert result['measures']['is_constant'] is True
        assert result['classification']['continuity'] == 'CONSTANT'
        assert result['classification']['temporal'] == 'CONSTANT'

    def test_trending_signal(self):
        values = np.linspace(0, 100, 2000)
        result = typology.from_observations(values)
        assert result['measures']['is_constant'] is False
        assert result['measures']['n_samples'] == 2000
        # Trending signal should have high hurst
        assert result['measures']['hurst'] > 0.6
        assert result['window']['window_size'] >= 64

    def test_random_signal(self):
        np.random.seed(42)
        values = np.random.randn(5000)
        result = typology.from_observations(values)
        assert result['measures']['is_constant'] is False
        pe = result['measures']['perm_entropy']
        assert pe > 0.8  # high entropy for random

    def test_periodic_signal(self):
        t = np.linspace(0, 20 * np.pi, 5000)
        values = np.sin(t)
        result = typology.from_observations(values)
        assert result['measures']['dominant_frequency'] is not None
        assert result['measures']['dominant_frequency'] > 0

    def test_empty_signal(self):
        values = np.array([])
        result = typology.from_observations(values)
        assert result['measures']['n_samples'] == 0
        assert result['measures']['is_constant'] is True

    def test_sparse_signal(self):
        values = np.zeros(1000)
        values[50] = 100.0
        values[500] = 200.0
        result = typology.from_observations(values)
        assert result['measures']['sparsity'] > 0.9


class TestFromFeatures:

    def test_trending_features(self):
        features = {
            'hurst': 0.99,
            'perm_entropy': 0.2,
            'signal_std': 50.0,
            'signal_mean': 100.0,
            'n_samples': 5000,
        }
        cls = typology.from_features(features)
        assert cls['temporal'] == 'TRENDING'
        assert cls['memory'] == 'LONG_MEMORY'
        assert cls['complexity'] == 'LOW'

    def test_constant_features(self):
        features = {
            'signal_std': 0.0,
            'signal_mean': 42.0,
        }
        cls = typology.from_features(features)
        assert cls['continuity'] == 'CONSTANT'
        assert cls['temporal'] == 'CONSTANT'

    def test_impulsive_features(self):
        features = {
            'kurtosis': 50.0,
            'crest_factor': 15.0,
            'signal_std': 10.0,
            'signal_mean': 0.0,
        }
        cls = typology.from_features(features)
        assert cls['temporal'] == 'IMPULSIVE'
        assert cls['distribution'] == 'HEAVY_TAILED'

    def test_stationary_features(self):
        features = {
            'adf_pvalue': 0.001,
            'kpss_pvalue': 0.10,  # >= 0.05 means KPSS says stationary
            'variance_ratio': 1.0,
            'hurst': 0.5,
            'perm_entropy': 0.6,
            'signal_std': 5.0,
            'signal_mean': 0.0,
            'n_samples': 5000,
        }
        cls = typology.from_features(features)
        assert cls['temporal'] == 'STATIONARY'
        assert cls['stationarity'] == 'STATIONARY'

    def test_all_10_dimensions_present(self):
        features = {
            'hurst': 0.5, 'perm_entropy': 0.5, 'signal_std': 1.0,
            'signal_mean': 0.0, 'n_samples': 1000,
        }
        cls = typology.from_features(features)
        expected_keys = [
            'continuity', 'stationarity', 'temporal', 'memory',
            'complexity', 'spectral', 'determinism', 'distribution',
            'amplitude', 'volatility',
        ]
        for key in expected_keys:
            assert key in cls, f"Missing dimension: {key}"


class TestConfig:

    def test_get_nested(self):
        val = typology.get_config('temporal.trending.hurst_strong')
        assert val == 0.99

    def test_get_caps(self):
        val = typology.get_config('caps.ftle.min')
        assert val == 500

    def test_get_missing(self):
        val = typology.get_config('nonexistent.path', default='NOPE')
        assert val == 'NOPE'


class TestSystemWindow:

    def test_max_method(self):
        from typology.window import system_window
        taus = [10.0, 50.0, 200.0, 30.0]
        ws = system_window(taus, method='max')
        assert ws == int(200.0 * 2.5)  # 500

    def test_empty_list(self):
        from typology.window import system_window
        ws = system_window([])
        assert ws == 64  # min_window


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
