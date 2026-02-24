"""
Mass Engine

Compute total energy in the system.
This is the MASS in Structure = Geometry × Mass.

Core metrics:
- total_variance: M = trace(Σ) = Σᵢ Var(xᵢ)
- energy_distribution: How energy is distributed across signals
- mass_trend: dM/dt (accumulating, depleting, stable)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class MassTrend(Enum):
    """Mass trend classification."""
    ACCUMULATING = "accumulating"
    DEPLETING = "depleting"
    STABLE = "stable"


@dataclass
class MassResult:
    """Result of mass computation."""
    total_variance: float           # M = trace(Σ)
    mean_variance: float            # M / N
    variance_dispersion: float      # std(Var) / mean(Var)
    energy_eff_dim: float          # Participation ratio of energy
    signal_variances: List[float]   # Per-signal variances
    n_signals: int


def compute_mass(X: np.ndarray) -> MassResult:
    """
    Compute mass (total variance) of signal matrix.

    Args:
        X: Signal matrix (n_timepoints × n_signals)

    Returns:
        MassResult with mass metrics
    """
    n_timepoints, n_signals = X.shape

    if n_signals < 1 or n_timepoints < 2:
        return MassResult(
            total_variance=0.0,
            mean_variance=0.0,
            variance_dispersion=0.0,
            energy_eff_dim=0.0,
            signal_variances=[],
            n_signals=n_signals,
        )

    # Compute per-signal variances
    variances = np.var(X, axis=0, ddof=1)
    variances = np.nan_to_num(variances, nan=0.0)

    # Total variance (mass)
    total_variance = np.sum(variances)

    # Mean variance
    mean_variance = np.mean(variances)

    # Variance dispersion (inequality)
    if mean_variance > 1e-10:
        variance_dispersion = np.std(variances) / mean_variance
    else:
        variance_dispersion = 0.0

    # Energy effective dimension
    if total_variance > 1e-10:
        p = variances / total_variance
        entropy = -np.sum(p * np.log(p + 1e-10))
        energy_eff_dim = np.exp(entropy)
    else:
        energy_eff_dim = 0.0

    return MassResult(
        total_variance=float(total_variance),
        mean_variance=float(mean_variance),
        variance_dispersion=float(variance_dispersion),
        energy_eff_dim=float(energy_eff_dim),
        signal_variances=variances.tolist(),
        n_signals=n_signals,
    )


def compute_mass_trajectory(
    masses: List[MassResult],
    trend_threshold: float = 0.01,
) -> Dict:
    """
    Analyze mass evolution over time.

    Args:
        masses: List of MassResult from sequential windows
        trend_threshold: Threshold for significant trend

    Returns:
        Dict with trajectory metrics including trend classification
    """
    if len(masses) < 2:
        return {
            'mass_trend': MassTrend.STABLE.value,
            'mass_slope': 0.0,
            'mass_slope_normalized': 0.0,
            'mass_min': 0.0,
            'mass_max': 0.0,
        }

    total_variances = np.array([m.total_variance for m in masses])
    t = np.arange(len(masses))

    # Compute trend
    slope = np.polyfit(t, total_variances, 1)[0] if len(t) > 1 else 0.0

    # Normalize by mean
    mean_mass = np.mean(total_variances)
    if mean_mass > 1e-10:
        slope_normalized = slope / mean_mass
    else:
        slope_normalized = 0.0

    # Classify trend
    if slope_normalized > trend_threshold:
        trend = MassTrend.ACCUMULATING
    elif slope_normalized < -trend_threshold:
        trend = MassTrend.DEPLETING
    else:
        trend = MassTrend.STABLE

    return {
        'mass_trend': trend.value,
        'mass_slope': float(slope),
        'mass_slope_normalized': float(slope_normalized),
        'mass_min': float(np.min(total_variances)),
        'mass_max': float(np.max(total_variances)),
        'mass_std': float(np.std(total_variances)),
    }
