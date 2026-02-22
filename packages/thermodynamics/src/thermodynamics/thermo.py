"""
Thermodynamic analogs from eigenvalue spectra.

Mapping:
    total_variance  (Σλᵢ)             →  Energy (E)
    Shannon entropy of λ distribution  →  Entropy (S)
    variance of eff_dim velocity       →  Temperature (T)
    E - T·S                            →  Free energy (F)

All computed from eigendecomp outputs. No new math.
"""

import numpy as np
from typing import Dict, Any, List


def compute_entropy(eigenvalues: np.ndarray) -> float:
    """Shannon entropy of eigenvalue distribution (bits)."""
    eig = np.asarray(eigenvalues, dtype=np.float64)
    eig = eig[eig > 0]
    if len(eig) == 0:
        return np.nan
    p = eig / np.sum(eig)
    return float(-np.sum(p * np.log(p + 1e-30)))


def compute_temperature(velocities: np.ndarray) -> float:
    """Effective temperature from velocity distribution (kT ~ var(v))."""
    v = np.asarray(velocities, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    return float(np.var(v))


def compute_thermodynamics(
    eigenvalues_sequence: List[np.ndarray],
    effective_dim_sequence: np.ndarray,
    total_variance_sequence: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute thermodynamic quantities from eigendecomp trajectory.

    Parameters
    ----------
    eigenvalues_sequence : list of np.ndarray
        Eigenvalues per window.
    effective_dim_sequence : np.ndarray
        Effective dimension per window.
    total_variance_sequence : np.ndarray
        Total variance per window.

    Returns
    -------
    dict with entropy, energy, temperature, free_energy, heat_capacity.
    """
    eff_dim = np.asarray(effective_dim_sequence, dtype=np.float64)
    total_var = np.asarray(total_variance_sequence, dtype=np.float64)

    # Entropy: mean Shannon entropy across windows
    entropies = [compute_entropy(eig) for eig in eigenvalues_sequence]
    valid_entropies = [e for e in entropies if np.isfinite(e)]
    entropy = float(np.mean(valid_entropies)) if valid_entropies else np.nan

    # Energy: mean total variance
    energy = float(np.nanmean(total_var))

    # Temperature: variance of eff_dim velocity
    velocities = np.diff(eff_dim)
    temperature = compute_temperature(velocities)

    # Free energy: F = E - T·S
    if np.isfinite(energy) and np.isfinite(temperature) and np.isfinite(entropy):
        free_energy = energy - temperature * entropy
    else:
        free_energy = np.nan

    # Heat capacity: dE/dT (finite difference over trajectory)
    energy_series = total_var[np.isfinite(total_var)]
    if len(energy_series) > 2:
        # Crude estimate: correlation of energy change with temperature proxy
        de = np.diff(energy_series)
        heat_capacity = float(np.var(de) / (temperature + 1e-30)) if np.isfinite(temperature) and temperature > 0 else np.nan
    else:
        heat_capacity = np.nan

    return {
        'entropy': entropy,
        'energy': energy,
        'temperature': temperature,
        'free_energy': free_energy,
        'heat_capacity': heat_capacity,
        'entropy_series': entropies,
        'n_windows': len(eigenvalues_sequence),
    }
