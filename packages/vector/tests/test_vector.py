"""Tests for vector package."""

import numpy as np
import pytest
import vector
from vector.registry import get_registry
from vector.signal import compute_signal
from vector.cohort import compute_cohort, pivot_to_matrix
from vector.system import compute_system


class TestRegistry:

    def test_discovers_44_engines(self):
        reg = get_registry()
        assert len(reg.engine_names) == 44

    def test_get_spec(self):
        reg = get_registry()
        spec = reg.get_spec('statistics')
        assert spec.base_window == 16
        assert spec.min_window == 4
        assert 'statistics_kurtosis' in spec.outputs

    def test_get_compute(self):
        reg = get_registry()
        func = reg.get_compute('statistics')
        result = func(np.random.randn(100))
        assert 'statistics_kurtosis' in result
        assert 'statistics_skewness' in result
        assert 'statistics_crest_factor' in result

    def test_unknown_engine(self):
        reg = get_registry()
        with pytest.raises(KeyError):
            reg.get_spec('nonexistent_engine')

    def test_group_by_window(self):
        reg = get_registry()
        groups = reg.group_by_window(['statistics', 'hurst', 'rqa'])
        # statistics=16, hurst=128, rqa=200 → 3 groups
        assert len(groups) == 3

    def test_all_yaml_outputs_match_engine(self):
        """Every YAML declares outputs. Every engine returns those keys."""
        reg = get_registry()
        y = np.random.randn(500)
        # Test engines that work with pure numpy (no pmtvs)
        safe_engines = [
            'statistics', 'trend', 'rate_of_change', 'peak', 'rms',
            'frequency_bands', 'transition_count', 'duty_cycle',
            'variance_ratio', 'variance_growth', 'pulsation_index',
            'level_count', 'level_histogram', 'dwell_times',
            'mean_time_between', 'cycle_counting', 'snr',
            'time_constant', 'spectral', 'entropy', 'transition_matrix',
        ]
        for name in safe_engines:
            func = reg.get_compute(name)
            result = func(y)
            expected = set(reg.get_outputs(name))
            actual = set(result.keys())
            assert expected.issubset(actual), \
                f"Engine {name}: missing keys {expected - actual}"


class TestSignal:

    def test_basic_windowed(self):
        """Compute signal vector with small set of engines."""
        y = np.random.randn(500)
        rows = compute_signal(
            signal_id='test',
            values=y,
            window_size=100,
            stride=50,
            engines=['statistics', 'trend'],
        )
        assert len(rows) > 0
        assert rows[0]['signal_id'] == 'test'
        assert rows[0]['window_index'] == 0
        assert 'statistics_kurtosis' in rows[0]
        assert 'trend_slope' in rows[0]

    def test_window_indices_sequential(self):
        y = np.random.randn(500)
        rows = compute_signal('s1', y, 100, 50, engines=['statistics'])
        indices = [r['window_index'] for r in rows]
        assert indices == list(range(len(indices)))

    def test_empty_signal(self):
        rows = compute_signal('empty', np.array([]), 100, 50, engines=['statistics'])
        assert rows == []

    def test_constant_signal(self):
        y = np.ones(200) * 42.0
        rows = compute_signal('const', y, 50, 25, engines=['statistics', 'peak'])
        assert len(rows) > 0
        assert rows[0]['statistics_kurtosis'] == 0.0
        assert rows[0]['peak_to_peak'] == 0.0

    def test_no_key_collisions(self):
        """Run multiple engines — no key should be overwritten."""
        y = np.random.randn(500)
        rows = compute_signal(
            'multi', y, 100, 50,
            engines=['statistics', 'trend', 'peak', 'rms', 'rate_of_change'],
        )
        # Each row should have keys from all engines
        row = rows[0]
        assert 'statistics_kurtosis' in row
        assert 'trend_slope' in row
        assert 'peak_value' in row
        assert 'rms_value' in row
        assert 'rate_of_change_mean' in row

    def test_short_signal_nan_fill(self):
        """Signal shorter than engine min_window → NaN fill."""
        y = np.random.randn(10)
        rows = compute_signal('short', y, 10, 10, engines=['rqa'])
        assert len(rows) > 0
        # rqa min_window=50, data is 10 → should be NaN
        assert np.isnan(rows[0]['rqa_recurrence_rate'])


class TestCohort:

    def test_centroid(self):
        # 3 signals, 4 features
        matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ])
        names = ['f1', 'f2', 'f3', 'f4']
        result = compute_cohort(matrix, 'c1', 0, names)
        assert result['cohort_id'] == 'c1'
        assert result['n_signals'] == 3
        assert result['centroid_f1'] == 2.0
        assert result['centroid_f4'] == 5.0

    def test_dispersion(self):
        matrix = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        names = ['x', 'y']
        result = compute_cohort(matrix, 'c1', 0, names)
        assert result['dispersion_mean'] > 0
        assert result['dispersion_max'] > 0

    def test_empty_cohort(self):
        matrix = np.zeros((0, 3))
        result = compute_cohort(matrix, 'empty', 0, ['a', 'b', 'c'])
        assert result['n_signals'] == 0
        assert np.isnan(result['centroid_a'])

    def test_pivot_to_matrix(self):
        rows = [
            {'signal_id': 's1', 'statistics_kurtosis': 1.0, 'trend_slope': 0.5},
            {'signal_id': 's2', 'statistics_kurtosis': 2.0, 'trend_slope': 0.3},
        ]
        features = ['statistics_kurtosis', 'trend_slope']
        m = pivot_to_matrix(rows, features)
        assert m.shape == (2, 2)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 0.3

    def test_nan_handling(self):
        matrix = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
        ])
        names = ['a', 'b']
        result = compute_cohort(matrix, 'c1', 0, names)
        # Should compute without error
        assert result['centroid_a'] == 2.0
        assert np.isfinite(result['centroid_b'])


class TestSystem:

    def test_system_centroid(self):
        matrix = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        names = ['f1', 'f2']
        result = compute_system(matrix, 0, names)
        assert result['n_cohorts'] == 3
        assert result['system_centroid_f1'] == 3.0
        assert result['system_centroid_f2'] == 4.0


class TestNamespacing:
    """Verify zero collisions in the full engine set."""

    def test_all_output_keys_unique(self):
        """No two engines declare the same output key."""
        reg = get_registry()
        seen = {}
        for name in reg.engine_names:
            for key in reg.get_outputs(name):
                if key in seen:
                    pytest.fail(
                        f"Key collision: '{key}' declared by both "
                        f"'{seen[key]}' and '{name}'"
                    )
                seen[key] = name

    def test_total_output_count(self):
        """Verify we have ~177 unique output keys."""
        reg = get_registry()
        all_keys = set()
        for name in reg.engine_names:
            all_keys.update(reg.get_outputs(name))
        assert len(all_keys) >= 170  # should be ~177


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
