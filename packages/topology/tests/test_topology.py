"""Tests for the topology package."""
import numpy as np
import pytest

class TestPersistence:
    def test_basic_output(self):
        from topology.homology import compute_persistence
        np.random.seed(42)
        cloud = np.random.randn(50, 3)
        result = compute_persistence(cloud)
        assert 'persistence_pairs' in result
        assert 'betti_0' in result
        assert result['n_points'] == 50

    def test_cluster_detection(self):
        from topology.homology import compute_persistence
        np.random.seed(42)
        c1 = np.random.randn(30, 2) + np.array([0, 0])
        c2 = np.random.randn(30, 2) + np.array([10, 10])
        cloud = np.vstack([c1, c2])
        result = compute_persistence(cloud)
        # Should detect structure (persistence > 0)
        assert result['total_persistence_0'] > 0

    def test_subsample(self):
        from topology.homology import compute_persistence
        cloud = np.random.randn(1000, 3)
        result = compute_persistence(cloud, max_points=100)
        assert result['n_points'] == 100

    def test_1d_input(self):
        from topology.homology import compute_persistence
        result = compute_persistence(np.random.randn(30))
        assert result['n_points'] == 30

    def test_too_short(self):
        from topology.homology import compute_persistence
        result = compute_persistence(np.array([[1, 2]]))
        assert result['n_points'] == 0

class TestBetti:
    def test_single_cluster(self):
        from topology.homology import betti_numbers_at_threshold
        cloud = np.random.randn(20, 2) * 0.1
        result = betti_numbers_at_threshold(cloud, threshold=1.0)
        assert result['betti_0'] == 1

    def test_two_clusters(self):
        from topology.homology import betti_numbers_at_threshold
        c1 = np.zeros((10, 2))
        c2 = np.ones((10, 2)) * 100
        cloud = np.vstack([c1, c2])
        result = betti_numbers_at_threshold(cloud, threshold=1.0)
        assert result['betti_0'] == 2
