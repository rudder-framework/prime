"""Tests for the pairwise package."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def correlated_signals():
    """Two highly correlated signals + one independent."""
    np.random.seed(42)
    base = np.random.randn(5)
    return np.array([
        base,                           # signal 0
        base * 0.9 + np.random.randn(5) * 0.1,  # signal 1 (correlated)
        np.random.randn(5),             # signal 2 (independent)
    ])


@pytest.fixture
def identical_signals():
    """All signals identical."""
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return np.array([v, v, v])


@pytest.fixture
def orthogonal_signals():
    """Signals along orthogonal axes."""
    return np.eye(4)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_correlated_pair(self, correlated_signals):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(
            correlated_signals[0], correlated_signals[1],
            signal_a='s0', signal_b='s1',
        )
        assert result['correlation'] > 0.8
        assert result['correlation_abs'] > 0.8
        assert result['distance'] < 2.0
        assert result['cosine_similarity'] > 0.8

    def test_independent_pair(self, correlated_signals):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(
            correlated_signals[0], correlated_signals[2],
            signal_a='s0', signal_b='s2',
        )
        # Should be less correlated
        assert abs(result['correlation']) < result['correlation_abs'] + 0.01

    def test_identical_vectors(self, identical_signals):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(
            identical_signals[0], identical_signals[1],
            signal_a='a', signal_b='b',
        )
        assert abs(result['correlation'] - 1.0) < 1e-10
        assert result['distance'] < 1e-10
        assert abs(result['cosine_similarity'] - 1.0) < 1e-10

    def test_output_keys(self, correlated_signals):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(
            correlated_signals[0], correlated_signals[1],
            signal_a='x', signal_b='y',
        )
        required_keys = ['signal_a', 'signal_b', 'correlation',
                         'correlation_abs', 'distance',
                         'cosine_similarity', 'mutual_info']
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_nan_handling(self):
        from pairwise.metrics import compute_pair_metrics
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = compute_pair_metrics(a, b)
        # Should compute on valid positions (indices 0, 3, 4)
        assert np.isfinite(result['correlation'])

    def test_short_vectors(self):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(np.array([1.0]), np.array([2.0]))
        assert np.isnan(result['correlation'])

    def test_with_centroid_context(self, correlated_signals):
        from pairwise.metrics import compute_pair_metrics_with_context
        centroid = np.mean(correlated_signals, axis=0)
        result = compute_pair_metrics_with_context(
            correlated_signals[0], correlated_signals[1],
            centroid, signal_a='s0', signal_b='s1',
        )
        assert 'dist_a_to_centroid' in result
        assert 'dist_b_to_centroid' in result
        assert 'same_side' in result
        assert np.isfinite(result['dist_a_to_centroid'])

    def test_mutual_info_nonnegative(self, correlated_signals):
        from pairwise.metrics import compute_pair_metrics
        result = compute_pair_metrics(
            correlated_signals[0], correlated_signals[1],
        )
        assert result['mutual_info'] >= 0


# ---------------------------------------------------------------------------
# Signal pairwise tests
# ---------------------------------------------------------------------------

class TestSignalPairwise:

    def test_pair_count(self, correlated_signals):
        from pairwise.signal import compute_signal_pairwise
        rows = compute_signal_pairwise(
            correlated_signals,
            signal_ids=['s0', 's1', 's2'],
        )
        # C(3,2) = 3 pairs
        assert len(rows) == 3

    def test_pair_count_4_signals(self):
        from pairwise.signal import compute_signal_pairwise
        np.random.seed(42)
        matrix = np.random.randn(4, 5)
        rows = compute_signal_pairwise(
            matrix, signal_ids=['a', 'b', 'c', 'd'],
        )
        # C(4,2) = 6
        assert len(rows) == 6

    def test_window_index_propagated(self, correlated_signals):
        from pairwise.signal import compute_signal_pairwise
        rows = compute_signal_pairwise(
            correlated_signals,
            signal_ids=['s0', 's1', 's2'],
            window_index=42,
        )
        for row in rows:
            assert row['I'] == 42

    def test_single_signal_returns_empty(self):
        from pairwise.signal import compute_signal_pairwise
        rows = compute_signal_pairwise(
            np.random.randn(1, 5),
            signal_ids=['only'],
        )
        assert len(rows) == 0

    def test_with_centroid(self, correlated_signals):
        from pairwise.signal import compute_signal_pairwise
        centroid = np.mean(correlated_signals, axis=0)
        rows = compute_signal_pairwise(
            correlated_signals,
            signal_ids=['s0', 's1', 's2'],
            centroid=centroid,
        )
        assert 'dist_a_to_centroid' in rows[0]

    def test_batch(self):
        from pairwise.signal import compute_signal_pairwise_batch
        np.random.seed(42)
        matrices = [np.random.randn(3, 4) for _ in range(5)]
        rows = compute_signal_pairwise_batch(
            matrices,
            signal_ids=['a', 'b', 'c'],
            window_indices=list(range(5)),
        )
        # 5 windows × C(3,2) = 5 × 3 = 15
        assert len(rows) == 15


# ---------------------------------------------------------------------------
# Cohort pairwise tests
# ---------------------------------------------------------------------------

class TestCohortPairwise:

    def test_cohort_pair_count(self):
        from pairwise.cohort import compute_cohort_pairwise
        np.random.seed(42)
        vectors = np.random.randn(4, 6)
        rows = compute_cohort_pairwise(
            vectors,
            cohort_ids=['u1', 'u2', 'u3', 'u4'],
        )
        # C(4,2) = 6
        assert len(rows) == 6

    def test_cohort_keys(self):
        from pairwise.cohort import compute_cohort_pairwise
        np.random.seed(42)
        vectors = np.random.randn(3, 4)
        rows = compute_cohort_pairwise(
            vectors,
            cohort_ids=['c1', 'c2', 'c3'],
        )
        assert 'cohort_a' in rows[0]
        assert 'cohort_b' in rows[0]
        assert 'signal_a' not in rows[0]  # renamed


# ---------------------------------------------------------------------------
# Coloading tests
# ---------------------------------------------------------------------------

class TestColoading:

    def test_coloading_flags_basic(self):
        from pairwise.coloading import compute_coloading_flags
        # 3 signals, 2 PCs
        # Signals 0 and 1 load heavily on PC0
        # Signal 2 loads on PC1
        loadings = np.array([
            [5.0, 0.1],   # signal 0: heavy on PC0
            [4.0, -0.2],  # signal 1: heavy on PC0
            [0.1, 6.0],   # signal 2: heavy on PC1
        ])
        flags = compute_coloading_flags(
            loadings, signal_ids=['s0', 's1', 's2'],
            threshold=0.3,
        )
        assert len(flags) == 3  # C(3,2) pairs

        # Find the s0-s1 pair
        s01 = [f for f in flags if set([f['signal_a'], f['signal_b']]) == {'s0', 's1'}][0]
        assert s01['needs_granger'] is True

        # s0-s2 should not need granger (different PCs)
        s02 = [f for f in flags if set([f['signal_a'], f['signal_b']]) == {'s0', 's2'}][0]
        assert s02['needs_granger'] is False

    def test_merge_coloading(self):
        from pairwise.coloading import compute_coloading_flags, merge_coloading_with_pairwise
        loadings = np.array([[3.0, 0.1], [2.0, 0.1], [0.1, 3.0]])
        coloading = compute_coloading_flags(loadings, ['a', 'b', 'c'], threshold=0.3)
        pairwise = [
            {'signal_a': 'a', 'signal_b': 'b', 'correlation': 0.9},
            {'signal_a': 'a', 'signal_b': 'c', 'correlation': 0.1},
            {'signal_a': 'b', 'signal_b': 'c', 'correlation': 0.2},
        ]
        merged = merge_coloading_with_pairwise(pairwise, coloading)
        assert merged[0]['needs_granger'] is True  # a-b high co-loading
        assert 'max_coloading' in merged[0]

    def test_no_signals(self):
        from pairwise.coloading import compute_coloading_flags
        flags = compute_coloading_flags(np.array([]).reshape(0, 3), [])
        assert len(flags) == 0
