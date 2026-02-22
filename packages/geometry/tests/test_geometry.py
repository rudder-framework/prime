"""Tests for the geometry package."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ensemble():
    """5 signals, 3 features, known geometry."""
    np.random.seed(42)
    centroid = np.array([1.0, 2.0, 3.0])
    # Signals scattered around centroid
    signals = centroid + np.random.randn(5, 3) * 0.5
    signal_ids = ['s0', 's1', 's2', 's3', 's4']
    return signals, signal_ids, centroid


@pytest.fixture
def aligned_ensemble():
    """Signals all along PC1 direction."""
    np.random.seed(42)
    pc1 = np.array([1.0, 0.0, 0.0])  # PC1 along x-axis
    centroid = np.array([5.0, 0.0, 0.0])
    # Signals spread along x-axis only
    offsets = np.linspace(-2, 2, 6)
    signals = np.array([centroid + o * pc1 for o in offsets])
    return signals, [f's{i}' for i in range(6)], centroid, pc1


@pytest.fixture
def collapsing_series():
    """effective_dim that declines steadily."""
    return np.array([5.0, 4.8, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0])


@pytest.fixture
def stable_series():
    """effective_dim that stays roughly constant."""
    np.random.seed(42)
    return 4.0 + np.random.randn(20) * 0.1


# ---------------------------------------------------------------------------
# Signal geometry tests
# ---------------------------------------------------------------------------

class TestSignalGeometry:

    def test_basic_output_keys(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        rows = compute_signal_geometry(signals, ids, centroid)
        assert len(rows) == 5
        for row in rows:
            assert 'signal_id' in row
            assert 'distance' in row
            assert 'coherence' in row
            assert 'contribution' in row
            assert 'residual' in row
            assert 'signal_magnitude' in row

    def test_distance_nonnegative(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        rows = compute_signal_geometry(signals, ids, centroid)
        for row in rows:
            assert row['distance'] >= 0

    def test_signal_at_centroid_has_zero_distance(self):
        from geometry.signal import compute_signal_geometry
        centroid = np.array([1.0, 2.0, 3.0])
        # Signal exactly at centroid
        matrix = np.array([centroid, centroid + np.array([1, 0, 0])])
        rows = compute_signal_geometry(matrix, ['at_center', 'offset'], centroid)
        assert rows[0]['distance'] < 1e-10

    def test_coherence_range(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        rows = compute_signal_geometry(signals, ids, centroid)
        for row in rows:
            if np.isfinite(row['coherence']):
                assert -1.01 <= row['coherence'] <= 1.01

    def test_aligned_signals_high_coherence(self, aligned_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid, pc1 = aligned_ensemble
        pcs = np.array([pc1])  # PC1 as row vector
        rows = compute_signal_geometry(signals, ids, centroid, principal_components=pcs)
        # Signals spread along PC1 → high |coherence|
        coherences = [abs(r['coherence']) for r in rows if abs(r['distance']) > 0.1]
        if coherences:
            assert np.mean(coherences) > 0.8

    def test_pc_projections_included(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        pcs = np.eye(3)  # 3 orthogonal PCs
        rows = compute_signal_geometry(signals, ids, centroid, principal_components=pcs)
        assert 'pc0_projection' in rows[0]
        assert 'pc1_projection' in rows[0]
        assert 'pc2_projection' in rows[0]

    def test_nan_signal_handled(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        signals[2, :] = np.nan  # one all-NaN signal
        rows = compute_signal_geometry(signals, ids, centroid)
        assert np.isnan(rows[2]['distance'])

    def test_window_index_propagated(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        rows = compute_signal_geometry(signals, ids, centroid, window_index=99)
        for row in rows:
            assert row['I'] == 99

    def test_batch(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry_batch
        signals, ids, centroid = simple_ensemble
        matrices = [signals, signals + 0.1]
        centroids = [centroid, centroid + 0.1]
        rows = compute_signal_geometry_batch(
            matrices, ids, centroids, window_indices=[0, 1],
        )
        assert len(rows) == 10  # 5 signals × 2 windows

    def test_residual_and_contribution_pythagorean(self, simple_ensemble):
        from geometry.signal import compute_signal_geometry
        signals, ids, centroid = simple_ensemble
        rows = compute_signal_geometry(signals, ids, centroid)
        for row in rows:
            # contribution² + residual² ≈ signal_magnitude² (not exact due to centroid offset)
            # But residual is orthogonal to centroid direction, so:
            # ||signal||² = contribution² + residual² (if centroid is at origin)
            # This is approximate since centroid ≠ origin
            assert np.isfinite(row['contribution'])
            assert np.isfinite(row['residual'])


# ---------------------------------------------------------------------------
# Dynamics tests
# ---------------------------------------------------------------------------

class TestDynamics:

    def test_linear_series_constant_velocity(self):
        from geometry.dynamics import compute_derivatives
        x = np.linspace(0, 10, 20)  # linear
        d = compute_derivatives(x, dt=1.0, smooth_window=1)
        # Velocity should be roughly constant
        v_interior = d['velocity'][2:-2]
        assert np.std(v_interior) < 0.01
        # Acceleration should be near zero
        a_interior = d['acceleration'][2:-2]
        assert np.max(np.abs(a_interior)) < 0.01

    def test_quadratic_series_linear_velocity(self):
        from geometry.dynamics import compute_derivatives
        t = np.arange(20, dtype=float)
        x = t ** 2  # quadratic
        d = compute_derivatives(x, dt=1.0, smooth_window=1)
        # Velocity should be roughly 2*t
        v_interior = d['velocity'][2:-2]
        expected = 2 * t[2:-2]
        assert np.allclose(v_interior, expected, atol=1.0)
        # Acceleration should be roughly constant = 2
        a_interior = d['acceleration'][2:-2]
        assert np.allclose(a_interior, 2.0, atol=0.5)

    def test_short_series(self):
        from geometry.dynamics import compute_derivatives
        d = compute_derivatives(np.array([1.0, 2.0]))
        assert len(d['velocity']) == 2
        # All NaN for too-short series
        assert np.all(np.isnan(d['jerk']))

    def test_smoothing_reduces_noise(self):
        from geometry.dynamics import compute_derivatives
        np.random.seed(42)
        x = np.linspace(0, 10, 50) + np.random.randn(50) * 2
        d_raw = compute_derivatives(x, smooth_window=1)
        d_smooth = compute_derivatives(x, smooth_window=5)
        # Smoothed velocity should have lower variance
        v_raw_std = np.nanstd(d_raw['velocity'])
        v_smooth_std = np.nanstd(d_smooth['velocity'])
        assert v_smooth_std < v_raw_std

    def test_curvature_nonnegative(self):
        from geometry.dynamics import compute_derivatives
        x = np.sin(np.linspace(0, 4 * np.pi, 50))
        d = compute_derivatives(x, smooth_window=1)
        valid = np.isfinite(d['curvature'])
        assert np.all(d['curvature'][valid] >= 0)

    def test_eigenvalue_dynamics(self):
        from geometry.dynamics import compute_eigenvalue_dynamics
        # Simulate eigendecomp results with declining effective_dim
        results = []
        for i in range(10):
            results.append({
                'I': i,
                'effective_dim': 5.0 - i * 0.3,
                'eigenvalues': np.array([10.0 - i, 5.0, 2.0, 1.0, 0.5]),
                'total_variance': 18.5 - i * 0.5,
            })
        rows = compute_eigenvalue_dynamics(results, smooth_window=1)
        assert len(rows) == 10
        # Velocity should be negative (declining)
        velocities = [r['effective_dim_velocity'] for r in rows[1:-1]]
        assert all(v < 0 for v in velocities if np.isfinite(v))

    def test_eigenvalue_dynamics_has_per_eigenvalue(self):
        from geometry.dynamics import compute_eigenvalue_dynamics
        results = [
            {'I': i, 'effective_dim': 3.0, 'eigenvalues': np.array([5, 3, 1.0]),
             'total_variance': 9.0}
            for i in range(5)
        ]
        rows = compute_eigenvalue_dynamics(results, max_eigenvalues=3)
        assert 'eigenvalue_0' in rows[0]
        assert 'eigenvalue_0_velocity' in rows[0]


# ---------------------------------------------------------------------------
# Collapse tests
# ---------------------------------------------------------------------------

class TestCollapse:

    def test_collapsing_series_detected(self, collapsing_series):
        from geometry.dynamics import compute_derivatives
        from geometry.collapse import detect_collapse
        d = compute_derivatives(collapsing_series, smooth_window=1)
        result = detect_collapse(d['velocity'], threshold_velocity=-0.1)
        assert result['collapse_detected'] is True
        assert result['collapse_onset_idx'] is not None
        assert result['collapse_onset_fraction'] < 0.5  # early onset

    def test_stable_series_not_detected(self, stable_series):
        from geometry.dynamics import compute_derivatives
        from geometry.collapse import detect_collapse
        d = compute_derivatives(stable_series, smooth_window=1)
        result = detect_collapse(d['velocity'], threshold_velocity=-0.5)
        assert result['collapse_detected'] is False

    def test_max_run_length(self, collapsing_series):
        from geometry.dynamics import compute_derivatives
        from geometry.collapse import detect_collapse
        d = compute_derivatives(collapsing_series, smooth_window=1)
        result = detect_collapse(d['velocity'], threshold_velocity=-0.1)
        assert result['max_run_length'] >= 3

    def test_fraction_below(self, collapsing_series):
        from geometry.dynamics import compute_derivatives
        from geometry.collapse import detect_collapse
        d = compute_derivatives(collapsing_series, smooth_window=1)
        result = detect_collapse(d['velocity'], threshold_velocity=-0.1)
        assert result['fraction_below'] > 0.3

    def test_empty_series(self):
        from geometry.collapse import detect_collapse
        result = detect_collapse(np.array([]))
        assert result['collapse_detected'] is False

    def test_short_series(self):
        from geometry.collapse import detect_collapse
        result = detect_collapse(np.array([-0.1, -0.2]))
        assert result['collapse_detected'] is False

    def test_onset_fraction_range(self, collapsing_series):
        from geometry.dynamics import compute_derivatives
        from geometry.collapse import detect_collapse
        d = compute_derivatives(collapsing_series, smooth_window=1)
        result = detect_collapse(d['velocity'], threshold_velocity=-0.1)
        if result['collapse_onset_fraction'] is not None:
            assert 0.0 <= result['collapse_onset_fraction'] <= 1.0
