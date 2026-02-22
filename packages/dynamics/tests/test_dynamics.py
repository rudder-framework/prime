"""Tests for the dynamics package."""
import numpy as np
import pytest


class TestFTLE:
    def test_basic_output_keys(self):
        from dynamics.ftle import compute_ftle
        np.random.seed(42)
        values = np.cumsum(np.random.randn(500))
        result = compute_ftle(values, min_samples=100)
        for key in ['ftle', 'confidence', 'n_samples', 'embedding_dim', 'embedding_tau', 'method', 'direction']:
            assert key in result

    def test_too_short_returns_nan(self):
        from dynamics.ftle import compute_ftle
        result = compute_ftle(np.array([1.0, 2.0, 3.0]), min_samples=100)
        assert np.isnan(result['ftle'])
        assert result['n_samples'] == 0

    def test_capped_at_max_samples(self):
        from dynamics.ftle import compute_ftle
        values = np.random.randn(5000)
        result = compute_ftle(values, min_samples=100)
        assert result['n_samples'] <= 2000

    def test_backward_direction(self):
        from dynamics.ftle import compute_ftle
        np.random.seed(42)
        values = np.cumsum(np.random.randn(500))
        result = compute_ftle(values, direction='backward', min_samples=100)
        assert result['direction'] == 'backward'

    def test_rolling_ftle(self):
        from dynamics.ftle import compute_ftle_rolling
        np.random.seed(42)
        values = np.cumsum(np.random.randn(2000))
        results = compute_ftle_rolling(values, window_size=500, stride=200, min_samples=100)
        assert len(results) > 0
        assert 'I' in results[0]
        assert 'window_start' in results[0]

    def test_rolling_ftle_short_signal(self):
        from dynamics.ftle import compute_ftle_rolling
        results = compute_ftle_rolling(np.random.randn(100), window_size=500)
        assert len(results) == 0

    def test_nan_values_handled(self):
        from dynamics.ftle import compute_ftle
        values = np.random.randn(500)
        values[100:150] = np.nan
        result = compute_ftle(values, min_samples=100)
        assert result['n_samples'] > 0
