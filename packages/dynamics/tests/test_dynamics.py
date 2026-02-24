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


class TestCacheStrategy:
    def test_cache_strategy_none(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(None)
        assert s['mode'] == 'lock'
        assert s['refresh_interval'] == 0

    def test_cache_strategy_adaptive(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(0.6)
        assert s['mode'] == 'adaptive'
        assert s['refresh_after_pct'] == 0.6
        assert s['refresh_interval'] == 5

    def test_cache_strategy_early_onset(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(0.1)
        assert s['mode'] == 'refresh_late'
        assert s['refresh_interval'] == 10

    def test_cache_strategy_very_late(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(0.99)
        assert s['mode'] == 'lock'

    def test_cache_strategy_boundary_020(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(0.2)
        assert s['mode'] == 'adaptive'

    def test_cache_strategy_boundary_095(self):
        from dynamics.ftle import get_cache_strategy
        s = get_cache_strategy(0.95)
        assert s['mode'] == 'adaptive'


class TestRollingFTLECaching:
    def test_rolling_ftle_with_d2_onset(self):
        from dynamics.ftle import compute_ftle_rolling
        np.random.seed(42)
        values = np.cumsum(np.random.randn(2000))
        results = compute_ftle_rolling(
            values, window_size=500, stride=200,
            min_samples=100, d2_onset_pct=0.5,
        )
        assert len(results) > 0
        assert 'I' in results[0]
        assert 'embedding_dim' in results[0]
        assert 'embedding_tau' in results[0]

    def test_rolling_ftle_backward_compat(self):
        """Without d2_onset_pct, should still work (lock mode)."""
        from dynamics.ftle import compute_ftle_rolling
        np.random.seed(42)
        values = np.cumsum(np.random.randn(2000))
        results = compute_ftle_rolling(
            values, window_size=500, stride=200, min_samples=100,
        )
        assert len(results) > 0
        for r in results:
            for key in ['ftle', 'confidence', 'n_samples', 'embedding_dim',
                        'embedding_tau', 'method', 'direction']:
                assert key in r

    def test_rolling_ftle_cached_dims_consistent(self):
        """In lock mode, all windows should share the same dim/tau."""
        from dynamics.ftle import compute_ftle_rolling
        np.random.seed(42)
        values = np.cumsum(np.random.randn(2000))
        results = compute_ftle_rolling(
            values, window_size=500, stride=200,
            min_samples=100, d2_onset_pct=None,
        )
        assert len(results) >= 2
        dims = [r['embedding_dim'] for r in results]
        taus = [r['embedding_tau'] for r in results]
        assert all(d == dims[0] for d in dims), f"dims varied in lock mode: {dims}"
        assert all(t == taus[0] for t in taus), f"taus varied in lock mode: {taus}"
