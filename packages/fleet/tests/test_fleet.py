"""Tests for the fleet package."""
import numpy as np
import pytest

class TestFleetEigendecomp:
    def test_basic_output(self):
        from fleet.analysis import compute_fleet_eigendecomp
        np.random.seed(42)
        centroids = {f'u{i}': np.random.randn(5) for i in range(10)}
        result = compute_fleet_eigendecomp(centroids)
        assert result['n_cohorts'] == 10
        assert result['effective_dim'] > 0

    def test_two_cohorts(self):
        from fleet.analysis import compute_fleet_eigendecomp
        centroids = {'u1': np.array([1, 0, 0.0]), 'u2': np.array([0, 1, 0.0])}
        result = compute_fleet_eigendecomp(centroids)
        assert result['n_cohorts'] == 2

    def test_single_cohort(self):
        from fleet.analysis import compute_fleet_eigendecomp
        result = compute_fleet_eigendecomp({'u1': np.array([1, 2, 3.0])})
        assert result['n_cohorts'] == 0

class TestFleetPairwise:
    def test_pair_count(self):
        from fleet.analysis import compute_fleet_pairwise
        centroids = {f'u{i}': np.random.randn(5) for i in range(4)}
        results = compute_fleet_pairwise(centroids)
        assert len(results) == 6  # C(4,2)

    def test_distance_nonneg(self):
        from fleet.analysis import compute_fleet_pairwise
        centroids = {f'u{i}': np.random.randn(5) for i in range(3)}
        results = compute_fleet_pairwise(centroids)
        assert all(r['distance'] >= 0 for r in results)

class TestFleetVelocity:
    def test_basic(self):
        from fleet.analysis import compute_fleet_velocity
        series = {'u1': [np.array([i, i*2, i*3.0]) for i in range(10)]}
        results = compute_fleet_velocity(series)
        assert len(results) == 9
        assert all(r['fleet_speed'] >= 0 for r in results)
