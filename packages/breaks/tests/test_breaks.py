"""Tests for the breaks package."""
import numpy as np
import pytest

class TestCUSUM:
    def test_clear_break(self):
        from breaks.detection import detect_breaks_cusum
        np.random.seed(42)
        before = np.random.randn(200) + 0
        after = np.random.randn(200) + 3  # mean shift of 3
        values = np.concatenate([before, after])
        result = detect_breaks_cusum(values)
        assert result['break_detected'] is True
        assert abs(result['break_index'] - 200) < 50

    def test_no_break(self):
        from breaks.detection import detect_breaks_cusum
        np.random.seed(42)
        values = np.random.randn(400)
        result = detect_breaks_cusum(values, threshold_sigma=5.0)
        # No clear break in iid noise at high threshold
        assert result['cusum_significance'] < 5.0

    def test_too_short(self):
        from breaks.detection import detect_breaks_cusum
        result = detect_breaks_cusum(np.array([1, 2, 3.0]))
        assert result['break_detected'] is False

    def test_constant_signal(self):
        from breaks.detection import detect_breaks_cusum
        result = detect_breaks_cusum(np.ones(100))
        assert result['break_detected'] is False

class TestPettitt:
    def test_clear_break(self):
        from breaks.detection import detect_breaks_pettitt
        np.random.seed(42)
        before = np.random.randn(200)
        after = np.random.randn(200) + 3
        values = np.concatenate([before, after])
        result = detect_breaks_pettitt(values)
        assert result['break_detected'] is True
        assert result['break_index'] is not None

    def test_p_value_range(self):
        from breaks.detection import detect_breaks_pettitt
        np.random.seed(42)
        values = np.random.randn(200)
        result = detect_breaks_pettitt(values)
        assert 0 <= result['p_value'] <= 1.0

    def test_too_short(self):
        from breaks.detection import detect_breaks_pettitt
        result = detect_breaks_pettitt(np.array([1.0, 2.0]))
        assert result['break_detected'] is False
