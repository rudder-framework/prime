"""Tests for the baseline package."""
import numpy as np
import pytest

class TestFleetBaseline:
    def test_basic_output(self):
        from baseline.reference import compute_fleet_baseline
        np.random.seed(42)
        cohorts = {
            'unit_1': np.random.randn(50, 5),
            'unit_2': np.random.randn(50, 5),
            'unit_3': np.random.randn(50, 5),
        }
        result = compute_fleet_baseline(cohorts)
        assert result['n_cohorts'] == 3
        assert result['centroid'].shape == (5,)
        assert len(result['eigenvalues']) == 5
        assert result['effective_dim'] > 0

    def test_early_fraction(self):
        from baseline.reference import compute_fleet_baseline
        np.random.seed(42)
        cohorts = {'u1': np.random.randn(100, 3)}
        result = compute_fleet_baseline(cohorts, early_fraction=0.1)
        assert result['n_pooled_windows'] == 10  # 10% of 100

    def test_empty_input(self):
        from baseline.reference import compute_fleet_baseline
        result = compute_fleet_baseline({})
        assert result['n_cohorts'] == 0

    def test_observation_departure(self):
        from baseline.reference import compute_fleet_baseline, compute_observation_departure
        np.random.seed(42)
        cohorts = {'u1': np.random.randn(50, 3), 'u2': np.random.randn(50, 3)}
        baseline = compute_fleet_baseline(cohorts)
        obs = np.array([10.0, 10.0, 10.0])  # far from baseline
        dep = compute_observation_departure(obs, baseline)
        assert dep['centroid_distance'] > 1.0

class TestSegmentComparison:
    def test_declining_system(self):
        from baseline.segments import compute_segment_comparison
        eff_dim = np.linspace(5.0, 2.0, 100)  # declining
        eig0 = np.linspace(3.0, 6.0, 100)  # concentrating
        total_var = np.ones(100) * 10.0
        result = compute_segment_comparison(eff_dim, eig0, total_var)
        assert result['effective_dim_delta'] < 0  # decreased
        assert result['eigenvalue_0_delta'] > 0  # concentrated

    def test_too_short(self):
        from baseline.segments import compute_segment_comparison
        result = compute_segment_comparison(np.array([1.0]), np.array([1.0]), np.array([1.0]))
        assert np.isnan(result['effective_dim_delta'])
