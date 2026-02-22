"""Tests for the eigendecomp package."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_signals():
    """5 signals, 3 features, signals lie along axes → equal eigenvalues."""
    np.random.seed(42)
    return np.random.randn(5, 3)


@pytest.fixture
def rank1_signals():
    """10 signals that all lie on a single line → effective_dim ≈ 1."""
    np.random.seed(42)
    t = np.linspace(0, 1, 10)
    # All signals are scaled versions of [1, 2, 3]
    direction = np.array([1.0, 2.0, 3.0])
    return np.outer(t, direction) + np.random.randn(10, 3) * 0.01


@pytest.fixture
def rank2_signals():
    """10 signals in a 2D plane embedded in 5D → effective_dim ≈ 2."""
    np.random.seed(42)
    n = 10
    # Two independent directions
    d1 = np.array([1, 0, 0, 0, 0], dtype=float)
    d2 = np.array([0, 1, 0, 0, 0], dtype=float)
    c1 = np.random.randn(n)
    c2 = np.random.randn(n)
    return np.outer(c1, d1) + np.outer(c2, d2) + np.random.randn(n, 5) * 0.01


@pytest.fixture
def constant_signals():
    """All signals identical → zero variance."""
    return np.ones((5, 4))


# ---------------------------------------------------------------------------
# Core eigendecomp tests
# ---------------------------------------------------------------------------

class TestComputeEigendecomp:

    def test_basic_output_keys(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals)
        expected_keys = [
            'eigenvalues', 'explained_ratio', 'total_variance',
            'effective_dim', 'effective_dim_entropy',
            'eigenvalue_entropy', 'eigenvalue_entropy_normalized',
            'condition_number', 'ratio_2_1', 'ratio_3_1',
            'energy_concentration', 'principal_components',
            'signal_loadings', 'n_signals', 'n_features',
            'n_features_valid', 'varying_mask',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_rank1_effective_dim(self, rank1_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(rank1_signals)
        assert result['effective_dim'] < 1.5, \
            f"Rank-1 signals should have effective_dim near 1, got {result['effective_dim']}"

    def test_rank2_effective_dim(self, rank2_signals):
        from eigendecomp.decompose import compute_eigendecomp
        # With zscore normalization, noise dimensions get amplified
        # Use no normalization to test raw rank structure
        result = compute_eigendecomp(rank2_signals, norm_method="none")
        assert 1.5 < result['effective_dim'] < 2.5, \
            f"Rank-2 signals should have effective_dim near 2, got {result['effective_dim']}"

    def test_explained_ratio_sums_to_one(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals)
        total = np.sum(result['explained_ratio'])
        assert abs(total - 1.0) < 1e-10, f"Explained ratios should sum to 1, got {total}"

    def test_eigenvalues_descending(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals)
        eigs = result['eigenvalues']
        for i in range(len(eigs) - 1):
            if np.isfinite(eigs[i]) and np.isfinite(eigs[i + 1]):
                assert eigs[i] >= eigs[i + 1] - 1e-10

    def test_constant_signals_returns_empty(self, constant_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(constant_signals)
        assert result['n_signals'] == 0 or np.isnan(result['effective_dim'])

    def test_too_few_signals(self):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(np.array([[1, 2, 3]]))
        assert result['n_signals'] == 0
        assert np.isnan(result['effective_dim'])

    def test_nan_rows_excluded(self):
        from eigendecomp.decompose import compute_eigendecomp
        np.random.seed(42)
        matrix = np.random.randn(10, 3)
        matrix[3, :] = np.nan  # one bad row
        matrix[7, :] = np.nan  # another bad row
        result = compute_eigendecomp(matrix)
        assert result['n_signals'] == 8

    def test_no_normalization(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals, norm_method="none")
        assert result['n_signals'] > 0
        assert np.isfinite(result['effective_dim'])

    def test_ratio_2_1_multimode_indicator(self, rank2_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(rank2_signals)
        # For rank-2 data, λ₂ should be comparable to λ₁
        assert result['ratio_2_1'] > 0.1, "Rank-2 data should have substantial ratio_2_1"

    def test_energy_concentration(self, rank1_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(rank1_signals)
        # Rank-1 data: λ₁ dominates
        assert result['energy_concentration'] > 0.8, \
            f"Rank-1 should have high energy concentration, got {result['energy_concentration']}"

    def test_signal_loadings_shape(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals)
        loadings = result['signal_loadings']
        assert loadings is not None
        assert loadings.shape[0] == result['n_signals']

    def test_principal_components_shape(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        result = compute_eigendecomp(identity_signals)
        pcs = result['principal_components']
        assert pcs is not None
        assert pcs.shape[0] == result['n_features_valid']
        assert pcs.shape[1] == result['n_features_valid']


# ---------------------------------------------------------------------------
# Continuity tests
# ---------------------------------------------------------------------------

class TestContinuity:

    def test_flip_detection(self):
        from eigendecomp.continuity import enforce_eigenvector_continuity
        prev = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        curr = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)  # PC0 and PC2 flipped
        corrected = enforce_eigenvector_continuity(curr, prev)
        assert np.allclose(corrected[0], [1, 0, 0])  # flipped back
        assert np.allclose(corrected[1], [0, 1, 0])  # unchanged
        assert np.allclose(corrected[2], [0, 0, 1])  # flipped back

    def test_no_previous_returns_current(self):
        from eigendecomp.continuity import enforce_eigenvector_continuity
        curr = np.eye(3)
        assert np.allclose(enforce_eigenvector_continuity(curr, None), curr)

    def test_shape_mismatch_returns_current(self):
        from eigendecomp.continuity import enforce_eigenvector_continuity
        curr = np.eye(3)
        prev = np.eye(4)
        assert np.allclose(enforce_eigenvector_continuity(curr, prev), curr)

    def test_1d_vector(self):
        from eigendecomp.continuity import enforce_eigenvector_continuity
        curr = np.array([-1.0, -2.0, -3.0])
        prev = np.array([1.0, 2.0, 3.0])
        corrected = enforce_eigenvector_continuity(curr, prev)
        assert np.allclose(corrected, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Jackknife tests
# ---------------------------------------------------------------------------

class TestJackknife:

    def test_basic_jackknife(self, rank2_signals):
        from eigendecomp.bootstrap import jackknife_effective_dim
        result = jackknife_effective_dim(rank2_signals)
        assert np.isfinite(result['effective_dim'])
        assert result['ci_lower'] < result['effective_dim'] <= result['ci_upper']
        assert result['n_jackknife'] == len(rank2_signals)

    def test_too_few_signals(self):
        from eigendecomp.bootstrap import jackknife_effective_dim
        result = jackknife_effective_dim(np.random.randn(2, 3))
        assert np.isnan(result['ci_lower'])

    def test_stable_estimate(self, rank1_signals):
        from eigendecomp.bootstrap import jackknife_effective_dim
        result = jackknife_effective_dim(rank1_signals)
        # For a clean rank-1 signal, jackknife should be tight
        width = result['ci_upper'] - result['ci_lower']
        assert width < 2.0, f"Jackknife CI too wide for rank-1: {width}"


# ---------------------------------------------------------------------------
# Flatten tests
# ---------------------------------------------------------------------------

class TestFlatten:

    def test_flatten_keys(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        from eigendecomp.flatten import flatten_result
        result = compute_eigendecomp(identity_signals)
        flat = flatten_result(result, max_eigenvalues=3)
        assert 'effective_dim' in flat
        assert 'eigenvalue_0' in flat
        assert 'explained_ratio_0' in flat
        assert 'cumulative_variance_0' in flat
        # No array values — all scalars
        for k, v in flat.items():
            assert isinstance(v, (int, float)), f"Key {k} has non-scalar value: {type(v)}"

    def test_flatten_with_window_index(self, identity_signals):
        from eigendecomp.decompose import compute_eigendecomp
        from eigendecomp.flatten import flatten_result
        result = compute_eigendecomp(identity_signals)
        result['I'] = 42
        flat = flatten_result(result)
        assert flat['I'] == 42


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

class TestBatch:

    def test_batch_sequence(self):
        from eigendecomp.decompose import compute_eigendecomp_batch
        np.random.seed(42)
        matrices = [np.random.randn(8, 4) for _ in range(5)]
        indices = list(range(5))
        results = compute_eigendecomp_batch(matrices, indices)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r['I'] == i
            assert np.isfinite(r['effective_dim'])

    def test_batch_continuity_applied(self):
        from eigendecomp.decompose import compute_eigendecomp_batch
        np.random.seed(42)
        # Create matrices where PCs are similar but might flip
        matrices = [np.random.randn(8, 3) for _ in range(3)]
        results = compute_eigendecomp_batch(matrices, [0, 1, 2], enforce_continuity=True)
        # Just verify it runs without error
        assert len(results) == 3
