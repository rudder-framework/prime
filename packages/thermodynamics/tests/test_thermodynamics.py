"""Tests for the thermodynamics package."""
import numpy as np
import pytest

class TestThermodynamics:
    def test_basic_output(self):
        from thermodynamics import compute_thermodynamics
        eigs = [np.array([5, 3, 1.0]) for _ in range(10)]
        eff_dim = np.linspace(3.0, 2.0, 10)
        total_var = np.ones(10) * 9.0
        result = compute_thermodynamics(eigs, eff_dim, total_var)
        for key in ['entropy', 'energy', 'temperature', 'free_energy']:
            assert key in result

    def test_entropy_positive(self):
        from thermodynamics.thermo import compute_entropy
        assert compute_entropy(np.array([5, 3, 1.0])) > 0

    def test_entropy_uniform_higher(self):
        from thermodynamics.thermo import compute_entropy
        uniform = compute_entropy(np.array([1.0, 1.0, 1.0]))
        peaked = compute_entropy(np.array([10.0, 0.1, 0.1]))
        assert uniform > peaked

    def test_temperature_from_velocity(self):
        from thermodynamics.thermo import compute_temperature
        assert compute_temperature(np.array([0.1, -0.1, 0.2, -0.2])) > 0

    def test_empty_eigenvalues(self):
        from thermodynamics.thermo import compute_entropy
        assert np.isnan(compute_entropy(np.array([])))
