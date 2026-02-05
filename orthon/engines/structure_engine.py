"""
Structure Engine

Combine geometry and mass into structural assessment.
Structure = Geometry × Mass

Core metrics:
- compression: C = M / eff_dim (energy per degree of freedom)
- absorption: A = (eff_dim - crisis) / (healthy - crisis)
- geometry_mass_coupling: r = corr(eff_dim, M)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from .signal_geometry import GeometryResult
from .mass_engine import MassResult


class StructureState(Enum):
    """Structure state classification."""
    RESILIENT = "resilient"
    STRESSED = "stressed"
    FRAGILE = "fragile"
    CRITICAL = "critical"


@dataclass
class StructureResult:
    """Result of structure computation."""
    compression: float              # M / eff_dim (energy per DOF)
    compression_rank: float         # Rank in historical distribution
    absorption: float               # Remaining capacity (0-1)
    geometry_mass_coupling: float   # Correlation between geometry and mass
    state: StructureState
    description: str


# Empirical thresholds (calibrated across domains)
EFF_DIM_CRISIS = 2.0    # Floor before collapse
EFF_DIM_HEALTHY = 6.0   # Ceiling for healthy state


def compute_structure(
    geometry: GeometryResult,
    mass: MassResult,
    historical_compressions: Optional[List[float]] = None,
) -> StructureResult:
    """
    Compute structural state from geometry and mass.

    Args:
        geometry: GeometryResult from geometry engine
        mass: MassResult from mass engine
        historical_compressions: Optional list of historical compression values

    Returns:
        StructureResult with structural assessment
    """
    # === COMPRESSION ===
    # C = M / eff_dim = energy per degree of freedom
    if geometry.eff_dim > 0:
        compression = mass.total_variance / geometry.eff_dim
    else:
        compression = np.inf

    # Rank in historical distribution
    if historical_compressions and len(historical_compressions) > 0:
        rank = np.mean(np.array(historical_compressions) < compression)
    else:
        rank = 0.5  # Unknown

    # === ABSORPTION CAPACITY ===
    # A = (eff_dim - crisis) / (healthy - crisis)
    if geometry.n_signals > 0:
        # Scale thresholds by system size
        scaled_healthy = min(EFF_DIM_HEALTHY, geometry.n_signals * 0.8)
        scaled_crisis = min(EFF_DIM_CRISIS, geometry.n_signals * 0.2)

        if scaled_healthy > scaled_crisis:
            absorption = (geometry.eff_dim - scaled_crisis) / (scaled_healthy - scaled_crisis)
            absorption = np.clip(absorption, 0, 1)
        else:
            absorption = 0.0
    else:
        absorption = 0.0

    # === STATE CLASSIFICATION ===
    if absorption > 0.7:
        state = StructureState.RESILIENT
        description = "Structure healthy. Energy well distributed."
    elif absorption > 0.4:
        state = StructureState.STRESSED
        description = "Structure stressed. Monitor compression."
    elif absorption > 0.2:
        state = StructureState.FRAGILE
        description = "Structure fragile. Approaching collapse threshold."
    else:
        state = StructureState.CRITICAL
        description = "CRITICAL: Near or past collapse threshold."

    return StructureResult(
        compression=float(compression) if np.isfinite(compression) else np.inf,
        compression_rank=float(rank),
        absorption=float(absorption),
        geometry_mass_coupling=0.0,  # Computed in trajectory analysis
        state=state,
        description=description,
    )


def compute_structure_trajectory(
    geometries: List[GeometryResult],
    masses: List[MassResult],
) -> Dict:
    """
    Analyze structure evolution over time.

    Args:
        geometries: List of GeometryResult from sequential windows
        masses: List of MassResult from sequential windows

    Returns:
        Dict with trajectory metrics including geometry-mass coupling
    """
    n = min(len(geometries), len(masses))
    if n < 3:
        return {
            'geometry_mass_coupling': 0.0,
            'compression_trend': 0.0,
            'absorption_trend': 0.0,
            'is_collapsing': False,
        }

    eff_dims = np.array([g.eff_dim for g in geometries[:n]])
    total_vars = np.array([m.total_variance for m in masses[:n]])

    # Geometry-mass coupling
    if np.std(eff_dims) > 1e-10 and np.std(total_vars) > 1e-10:
        coupling = np.corrcoef(eff_dims, total_vars)[0, 1]
        if not np.isfinite(coupling):
            coupling = 0.0
    else:
        coupling = 0.0

    # Compression over time
    compressions = total_vars / (eff_dims + 1e-10)
    t = np.arange(n)
    compression_trend = np.polyfit(t, compressions, 1)[0] if n > 1 else 0.0

    # Absorption over time
    absorptions = []
    for g in geometries[:n]:
        scaled_healthy = min(EFF_DIM_HEALTHY, g.n_signals * 0.8)
        scaled_crisis = min(EFF_DIM_CRISIS, g.n_signals * 0.2)
        if scaled_healthy > scaled_crisis:
            a = (g.eff_dim - scaled_crisis) / (scaled_healthy - scaled_crisis)
            absorptions.append(np.clip(a, 0, 1))
        else:
            absorptions.append(0.0)

    absorptions = np.array(absorptions)
    absorption_trend = np.polyfit(t, absorptions, 1)[0] if n > 1 else 0.0

    # Collapse detection
    is_collapsing = (
        absorption_trend < -0.01 or  # Declining absorption
        (coupling < -0.3 and np.mean(total_vars[-3:]) > np.mean(total_vars[:3]))  # Accumulation-style collapse
    )

    return {
        'geometry_mass_coupling': float(coupling),
        'compression_trend': float(compression_trend),
        'absorption_trend': float(absorption_trend),
        'is_collapsing': bool(is_collapsing),
        'mean_compression': float(np.mean(compressions)),
        'max_compression': float(np.max(compressions)),
    }


def interpret_coupling(coupling: float, system_type: str) -> Dict:
    """
    Interpret geometry-mass coupling based on system type.

    Args:
        coupling: Correlation coefficient between eff_dim and mass
        system_type: System type from typology engine

    Returns:
        Dict with interpretation
    """
    if system_type == "accumulation":
        # Expected: r < 0 (mass drives geometry collapse)
        if coupling < -0.3:
            interpretation = "EXPECTED: Mass accumulation compressing geometry"
            severity = 2  # Warning
        elif coupling > 0.3:
            interpretation = "UNEXPECTED: Positive coupling in accumulation system"
            severity = 3  # Attention
        else:
            interpretation = "Weak coupling in accumulation system"
            severity = 1

    elif system_type == "degradation":
        # Expected: r > 0 (both decline together)
        if coupling > 0.3:
            interpretation = "EXPECTED: Geometry and mass declining together"
            severity = 2  # Warning
        elif coupling < -0.3:
            interpretation = "UNEXPECTED: Negative coupling in degradation system"
            severity = 3  # Attention
        else:
            interpretation = "Weak coupling in degradation system"
            severity = 1

    elif system_type == "conservation":
        # Expected: r ≈ 0 (geometry independent of excitation)
        if abs(coupling) < 0.3:
            interpretation = "HEALTHY: Geometry independent of mass variations"
            severity = 0  # Normal
        else:
            interpretation = "WARNING: Resonance forming (geometry responding to mass)"
            severity = 3  # Danger

    else:
        interpretation = f"Coupling r={coupling:.2f} for unknown system type"
        severity = 1

    return {
        'interpretation': interpretation,
        'severity': severity,
        'coupling': coupling,
    }
