"""Tests for the velocity package."""
import numpy as np
import pytest


class TestVelocityField:
    def test_basic_output(self):
        from velocity.field import compute_velocity_field
        np.random.seed(42)
        matrix = np.cumsum(np.random.randn(20, 4), axis=0)
        rows = compute_velocity_field(matrix, ['a', 'b', 'c', 'd'])
        assert len(rows) == 18  # N-2
        for key in ['I', 'speed', 'curvature', 'dominant_motion_signal', 'motion_dimensionality']:
            assert key in rows[0]

    def test_speed_nonnegative(self):
        from velocity.field import compute_velocity_field
        matrix = np.cumsum(np.random.randn(50, 3), axis=0)
        rows = compute_velocity_field(matrix, ['x', 'y', 'z'])
        assert all(r['speed'] >= 0 for r in rows)

    def test_curvature_nonnegative(self):
        from velocity.field import compute_velocity_field
        matrix = np.cumsum(np.random.randn(50, 3), axis=0)
        rows = compute_velocity_field(matrix, ['x', 'y', 'z'])
        assert all(r['curvature'] >= 0 for r in rows)

    def test_linear_trajectory_low_curvature(self):
        from velocity.field import compute_velocity_field
        t = np.linspace(0, 10, 50)
        matrix = np.column_stack([t, 2*t, 3*t])
        rows = compute_velocity_field(matrix, ['x', 'y', 'z'])
        curvatures = [r['curvature'] for r in rows]
        assert np.mean(curvatures) < 0.01

    def test_motion_dimensionality_range(self):
        from velocity.field import compute_velocity_field
        matrix = np.cumsum(np.random.randn(50, 5), axis=0)
        rows = compute_velocity_field(matrix, ['a', 'b', 'c', 'd', 'e'])
        for r in rows:
            assert 1.0 <= r['motion_dimensionality'] <= 5.0

    def test_too_short(self):
        from velocity.field import compute_velocity_field
        rows = compute_velocity_field(np.random.randn(2, 3), ['a', 'b', 'c'])
        assert len(rows) == 0

    def test_dominant_signal_valid(self):
        from velocity.field import compute_velocity_field
        matrix = np.cumsum(np.random.randn(20, 3), axis=0)
        rows = compute_velocity_field(matrix, ['sensor1', 'sensor2', 'sensor3'])
        for r in rows:
            assert r['dominant_motion_signal'] in ['sensor1', 'sensor2', 'sensor3']

    def test_nan_handling(self):
        from velocity.field import compute_velocity_field
        matrix = np.cumsum(np.random.randn(30, 3), axis=0)
        matrix[10:15, 1] = np.nan
        rows = compute_velocity_field(matrix, ['a', 'b', 'c'])
        assert len(rows) > 0
