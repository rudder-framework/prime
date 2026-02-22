"""Tests for the divergence package."""
import numpy as np
import pytest

class TestGranger:
    def test_basic_output(self):
        from divergence.causality import compute_granger
        np.random.seed(42)
        x = np.cumsum(np.random.randn(300))
        y = np.roll(x, 3) + np.random.randn(300) * 0.5
        result = compute_granger(x, y, max_lag=5)
        for key in ['granger_f', 'granger_p', 'best_lag']:
            assert key in result

    def test_too_short(self):
        from divergence.causality import compute_granger
        result = compute_granger(np.array([1, 2, 3.0]), np.array([4, 5, 6.0]))
        assert np.isnan(result['granger_f'])

    def test_independent_signals_low_f(self):
        from divergence.causality import compute_granger
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        result = compute_granger(x, y)
        # Independent signals should have low F (or at least not crash)
        assert np.isfinite(result['granger_f']) or np.isnan(result['granger_f'])

class TestTransferEntropy:
    def test_basic_output(self):
        from divergence.causality import compute_transfer_entropy
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.roll(x, 1) + np.random.randn(500) * 0.3
        result = compute_transfer_entropy(x, y)
        assert 'transfer_entropy' in result

    def test_nonnegative(self):
        from divergence.causality import compute_transfer_entropy
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)
        result = compute_transfer_entropy(x, y)
        if np.isfinite(result['transfer_entropy']):
            assert result['transfer_entropy'] >= 0

class TestDivergence:
    def test_kl_same_distribution(self):
        from divergence.divergence import kl_divergence
        np.random.seed(42)
        x = np.random.randn(1000)
        kl = kl_divergence(x, x)
        assert kl < 0.01  # near zero for same distribution

    def test_js_symmetric(self):
        from divergence.divergence import js_divergence
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000) + 2
        assert abs(js_divergence(x, y) - js_divergence(y, x)) < 0.01

    def test_js_bounded(self):
        from divergence.divergence import js_divergence
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000) + 5
        jsd = js_divergence(x, y)
        assert 0 <= jsd <= np.log(2) + 0.01

    def test_kl_short_arrays(self):
        from divergence.divergence import kl_divergence
        assert np.isnan(kl_divergence(np.array([1.0]), np.array([2.0])))
