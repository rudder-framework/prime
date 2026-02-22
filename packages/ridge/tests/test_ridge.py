"""Tests for the ridge package."""
import numpy as np
import pytest

class TestRidgeProximity:
    def test_basic_output(self):
        from ridge.proximity import compute_ridge_proximity
        ftle = np.linspace(0.01, 0.1, 20)
        speed = np.ones(20) * 0.5
        rows = compute_ridge_proximity(ftle, speed)
        assert len(rows) == 20
        for key in ['I', 'ftle_current', 'ftle_gradient', 'speed', 'urgency', 'time_to_ridge']:
            assert key in rows[0]

    def test_approaching_ridge_positive_urgency(self):
        from ridge.proximity import compute_ridge_proximity
        ftle = np.linspace(0.01, 0.1, 20)  # increasing FTLE
        speed = np.ones(20) * 1.0
        rows = compute_ridge_proximity(ftle, speed)
        # Positive gradient + positive speed = positive urgency
        mid = rows[len(rows)//2]
        assert mid['urgency'] > 0

    def test_too_short(self):
        from ridge.proximity import compute_ridge_proximity
        rows = compute_ridge_proximity(np.array([0.1]), np.array([1.0]))
        assert len(rows) == 0

    def test_time_to_ridge_computed(self):
        from ridge.proximity import compute_ridge_proximity
        ftle = np.linspace(0.01, 0.04, 20)  # below threshold, approaching
        speed = np.ones(20)
        rows = compute_ridge_proximity(ftle, speed, ridge_threshold=0.05)
        finite_ttr = [r['time_to_ridge'] for r in rows if np.isfinite(r['time_to_ridge'])]
        assert len(finite_ttr) > 0

    def test_nan_handling(self):
        from ridge.proximity import compute_ridge_proximity
        ftle = np.ones(20) * 0.03
        ftle[5:8] = np.nan
        speed = np.ones(20)
        rows = compute_ridge_proximity(ftle, speed)
        assert len(rows) == 20
