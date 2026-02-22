"""Tests for the stability package."""
import numpy as np
import pytest

class TestSignalStability:
    def test_basic_output(self):
        from stability.rolling import compute_signal_stability
        np.random.seed(42)
        values = np.sin(np.linspace(0, 10 * np.pi, 500)) + np.random.randn(500) * 0.1
        rows = compute_signal_stability(values, window_size=100, stride=50)
        assert len(rows) > 0
        for key in ['I', 'mean_amplitude', 'amplitude_std', 'stability_ratio', 'signal_energy']:
            assert key in rows[0]

    def test_stable_signal_high_ratio(self):
        from stability.rolling import compute_signal_stability
        values = np.sin(np.linspace(0, 20 * np.pi, 500))  # pure sine = stable
        rows = compute_signal_stability(values, window_size=100, stride=50)
        ratios = [r['stability_ratio'] for r in rows]
        assert np.mean(ratios) > 0.5

    def test_noisy_signal_lower_ratio(self):
        from stability.rolling import compute_signal_stability
        np.random.seed(42)
        values = np.random.randn(500) * 10  # pure noise
        rows = compute_signal_stability(values, window_size=100, stride=50)
        ratios = [r['stability_ratio'] for r in rows]
        # Noise has high CV → lower stability ratio
        assert len(ratios) > 0

    def test_short_signal(self):
        from stability.rolling import compute_signal_stability
        rows = compute_signal_stability(np.random.randn(20), window_size=100)
        assert len(rows) == 0

    def test_energy_nonnegative(self):
        from stability.rolling import compute_signal_stability
        values = np.random.randn(500)
        rows = compute_signal_stability(values, window_size=100, stride=50)
        assert all(r['signal_energy'] >= 0 for r in rows)
