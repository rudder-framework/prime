"""
Level 0: System Typology Engine

Defines the analytical landscape. The same geometry metric means different
things in different systems.

Types:
- ACCUMULATION: Mass builds up, geometry compressed (markets, queues)
- DEGRADATION: Mass depletes, geometry collapses (turbofan, bone)
- CONSERVATION: Mass held constant, geometry contains (buildings, bridges)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass


class SystemType(Enum):
    """System typology based on mass-geometry relationship."""
    ACCUMULATION = "accumulation"   # Mass ↑, Geometry ↓ (compressed by mass)
    DEGRADATION = "degradation"     # Mass ↓, Geometry ↓ (collapses with mass)
    CONSERVATION = "conservation"   # Mass ≈ 0, Geometry stable (contains mass)
    UNKNOWN = "unknown"


@dataclass
class TypologyResult:
    """Result of system typology classification."""
    system_type: SystemType
    mass_trend: float               # Slope of mass over time
    geometry_trend: float           # Slope of eff_dim over time
    coupling: float                 # Correlation between mass and geometry
    confidence: float               # Classification confidence (0-1)
    description: str


def classify_system_type(
    mass_series: np.ndarray,
    geometry_series: np.ndarray,
    trend_threshold: float = 0.01,
    coupling_threshold: float = 0.3,
) -> TypologyResult:
    """
    Classify system type based on mass and geometry trends.

    Args:
        mass_series: Time series of total variance (mass)
        geometry_series: Time series of effective dimension (geometry)
        trend_threshold: Threshold for significant trend
        coupling_threshold: Threshold for significant coupling

    Returns:
        TypologyResult with classification
    """
    n = len(mass_series)
    if n < 3:
        return TypologyResult(
            system_type=SystemType.UNKNOWN,
            mass_trend=0.0,
            geometry_trend=0.0,
            coupling=0.0,
            confidence=0.0,
            description="Insufficient data for typology",
        )

    # Compute trends
    t = np.arange(n)
    mass_trend = _compute_trend(t, mass_series)
    geometry_trend = _compute_trend(t, geometry_series)

    # Compute coupling (correlation between mass and geometry)
    if np.std(mass_series) > 1e-10 and np.std(geometry_series) > 1e-10:
        coupling = np.corrcoef(mass_series, geometry_series)[0, 1]
    else:
        coupling = 0.0

    # Classify based on trends and coupling
    mass_increasing = mass_trend > trend_threshold
    mass_decreasing = mass_trend < -trend_threshold
    mass_stable = abs(mass_trend) <= trend_threshold

    geometry_decreasing = geometry_trend < -trend_threshold
    geometry_stable = abs(geometry_trend) <= trend_threshold

    # Decision logic
    if mass_increasing and geometry_decreasing:
        system_type = SystemType.ACCUMULATION
        description = "Mass accumulating, compressing geometry. Monitor for B-tipping."
        confidence = min(abs(mass_trend), abs(geometry_trend)) / trend_threshold

    elif mass_decreasing and geometry_decreasing:
        system_type = SystemType.DEGRADATION
        description = "Mass depleting with geometry collapse. Watch for R-tipping."
        confidence = min(abs(mass_trend), abs(geometry_trend)) / trend_threshold

    elif mass_stable and geometry_stable:
        system_type = SystemType.CONSERVATION
        description = "Mass conserved, geometry stable. Structural integrity maintained."
        confidence = 1.0 - max(abs(mass_trend), abs(geometry_trend)) / trend_threshold

    elif mass_stable and geometry_decreasing:
        system_type = SystemType.CONSERVATION
        description = "Mass stable but geometry degrading. Resonance possible."
        confidence = 0.7

    elif mass_increasing and geometry_stable:
        system_type = SystemType.ACCUMULATION
        description = "Mass accumulating, geometry absorbing. Pre-compression phase."
        confidence = 0.6

    else:
        system_type = SystemType.UNKNOWN
        description = "Ambiguous trend pattern. Further analysis needed."
        confidence = 0.3

    # Adjust confidence based on coupling
    if system_type == SystemType.ACCUMULATION and coupling < -coupling_threshold:
        confidence = min(confidence * 1.2, 1.0)  # Expected coupling confirms
    elif system_type == SystemType.DEGRADATION and coupling > coupling_threshold:
        confidence = min(confidence * 1.2, 1.0)  # Expected coupling confirms
    elif system_type == SystemType.CONSERVATION and abs(coupling) < coupling_threshold:
        confidence = min(confidence * 1.2, 1.0)  # Expected decoupling confirms

    return TypologyResult(
        system_type=system_type,
        mass_trend=float(mass_trend),
        geometry_trend=float(geometry_trend),
        coupling=float(coupling) if np.isfinite(coupling) else 0.0,
        confidence=float(np.clip(confidence, 0, 1)),
        description=description,
    )


def _compute_trend(t: np.ndarray, y: np.ndarray) -> float:
    """Compute normalized trend slope."""
    if len(t) < 2:
        return 0.0

    # Linear regression
    try:
        slope, _ = np.polyfit(t, y, 1)
    except Exception:
        return 0.0

    # Normalize by mean
    mean_y = np.mean(y)
    if abs(mean_y) > 1e-10:
        return slope / mean_y
    else:
        return slope


def interpret_typology_for_domain(
    typology: TypologyResult,
    domain: str,
) -> str:
    """
    Provide domain-specific interpretation of typology.

    Args:
        typology: TypologyResult from classify_system_type
        domain: Domain identifier (e.g., "markets", "turbofan", "building")

    Returns:
        Domain-specific interpretation string
    """
    t = typology.system_type

    if domain in ["markets", "finance", "trading"]:
        if t == SystemType.ACCUMULATION:
            return "Risk accumulating. Correlations rising. Market stress building."
        elif t == SystemType.DEGRADATION:
            return "Market correction in progress. Correlations unwinding."
        elif t == SystemType.CONSERVATION:
            return "Stable market conditions. Normal regime."

    elif domain in ["turbofan", "engine", "industrial"]:
        if t == SystemType.DEGRADATION:
            return "Equipment degradation in progress. No early warning available (R-tipping)."
        elif t == SystemType.CONSERVATION:
            return "Equipment operating normally. Structural integrity maintained."
        elif t == SystemType.ACCUMULATION:
            return "Unusual pattern - check for sensor drift or process change."

    elif domain in ["building", "structure", "bridge"]:
        if t == SystemType.CONSERVATION:
            return "Structural health normal. Vibration modes stable."
        elif t == SystemType.DEGRADATION:
            return "Structural degradation detected. Investigate foundation/joints."
        elif t == SystemType.ACCUMULATION:
            return "Load accumulation detected. Check for resonance conditions."

    return typology.description
