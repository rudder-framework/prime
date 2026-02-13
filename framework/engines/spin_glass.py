"""
Spin Glass Engine

Map eigenstructure metrics to statistical mechanics order parameters.
Provides physical interpretation via Parisi's replica symmetry framework.

Order parameters:
- q (overlap): 1 - eff_dim/N (degree of coupling)
- m (magnetization): (N₊ - N₋)/N (net trend direction)

Phases:
- PARAMAGNETIC: q≈0, m≈0 (healthy, independent)
- FERROMAGNETIC: q>0, m≠0 (trending, aligned)
- SPIN_GLASS: q>0, m≈0 (fragile, coupled but directionless)
- MIXED: q>0, m≠0, CSD (trend with regime uncertainty)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class SpinGlassPhase(Enum):
    """Spin glass phase classification."""
    PARAMAGNETIC = "paramagnetic"       # Healthy: signals independent
    FERROMAGNETIC = "ferromagnetic"     # Trending: signals aligned
    SPIN_GLASS = "spin_glass"           # Fragile: coupled but directionless
    MIXED = "mixed"                     # Complex: trend with uncertainty
    UNKNOWN = "unknown"


@dataclass
class SpinGlassResult:
    """Result of spin glass analysis."""
    phase: SpinGlassPhase
    overlap_q: float                # Edwards-Anderson order parameter
    magnetization_m: float          # Net trend direction
    absorption: float               # 1 - q = resilience
    frustration: float              # Fraction of frustrated triangles
    below_at_line: bool            # Below de Almeida-Thouless line
    resilience: str                # HIGH/MODERATE/LOW/CRITICAL
    shock_response: str            # absorbed/partial/propagation/amplification
    description: str


# Empirical thresholds (calibrated across domains)
Q_THRESHOLD = 0.3       # Overlap threshold for phase transition
M_THRESHOLD = 0.2       # Magnetization threshold for ferromagnetic
EFF_DIM_HEALTHY = 6.0   # Reference healthy eff_dim
EFF_DIM_CRISIS = 2.0    # Reference crisis eff_dim


def compute_spin_glass(
    geometry_eff_dim: float,
    n_signals: int,
    signal_trends: Optional[List[float]] = None,
    correlation_matrix: Optional[np.ndarray] = None,
    csd_detected: bool = False,
) -> SpinGlassResult:
    """
    Compute spin glass order parameters and phase classification.

    Args:
        geometry_eff_dim: Effective dimension from geometry engine
        n_signals: Number of signals
        signal_trends: Optional list of per-signal trends (slopes)
        correlation_matrix: Optional correlation matrix for frustration
        csd_detected: Whether CSD has been detected

    Returns:
        SpinGlassResult with phase classification and order parameters
    """
    if n_signals < 2:
        return _empty_spin_glass_result()

    # === OVERLAP q (Edwards-Anderson) ===
    # Theoretical: q = 1 - eff_dim/N
    # Empirically scaled:
    q_theoretical = 1 - geometry_eff_dim / n_signals
    q_theoretical = np.clip(q_theoretical, 0, 1)

    # Scaled version using empirical thresholds
    scaled_healthy = min(EFF_DIM_HEALTHY, n_signals * 0.8)
    scaled_crisis = min(EFF_DIM_CRISIS, n_signals * 0.2)

    if scaled_healthy > scaled_crisis:
        q_scaled = (scaled_healthy - geometry_eff_dim) / (scaled_healthy - scaled_crisis)
        q_scaled = np.clip(q_scaled, 0, 1)
    else:
        q_scaled = q_theoretical

    # Use scaled version as primary
    overlap_q = q_scaled

    # === MAGNETIZATION m ===
    if signal_trends and len(signal_trends) == n_signals:
        n_positive = sum(1 for t in signal_trends if t > 0)
        n_negative = sum(1 for t in signal_trends if t < 0)
        magnetization_m = (n_positive - n_negative) / n_signals
    else:
        magnetization_m = 0.0  # Unknown

    # === ABSORPTION (resilience) ===
    absorption = 1 - overlap_q

    # === FRUSTRATION ===
    if correlation_matrix is not None:
        frustration = _compute_frustration(correlation_matrix)
    else:
        frustration = 0.0  # Unknown

    # === AT LINE (phase transition boundary) ===
    # Simplified: below AT line when q > threshold AND high frustration
    below_at_line = overlap_q > Q_THRESHOLD and frustration > 0.1

    # === PHASE CLASSIFICATION ===
    if overlap_q < Q_THRESHOLD:
        if abs(magnetization_m) < M_THRESHOLD:
            phase = SpinGlassPhase.PARAMAGNETIC
            description = "HEALTHY: Signals independent, no collective behavior"
        else:
            phase = SpinGlassPhase.FERROMAGNETIC
            description = "TRENDING: Signals aligned but not coupled"
    else:
        if abs(magnetization_m) < M_THRESHOLD:
            phase = SpinGlassPhase.SPIN_GLASS
            description = "FRAGILE: Coupled but directionless (glass phase)"
        else:
            if csd_detected:
                phase = SpinGlassPhase.MIXED
                description = "CRITICAL: Trending with regime uncertainty (CSD detected)"
            else:
                phase = SpinGlassPhase.FERROMAGNETIC
                description = "ALIGNED: Signals coupled and trending together"

    # === RESILIENCE CLASSIFICATION ===
    if absorption > 0.7:
        resilience = "HIGH"
    elif absorption > 0.4:
        resilience = "MODERATE"
    elif absorption > 0.2:
        resilience = "LOW"
    else:
        resilience = "CRITICAL"

    # === SHOCK RESPONSE ===
    if phase == SpinGlassPhase.PARAMAGNETIC:
        shock_response = "absorbed"
    elif phase == SpinGlassPhase.FERROMAGNETIC and not csd_detected:
        shock_response = "partial"
    elif phase == SpinGlassPhase.SPIN_GLASS:
        shock_response = "propagation"
    else:
        shock_response = "amplification"

    return SpinGlassResult(
        phase=phase,
        overlap_q=float(overlap_q),
        magnetization_m=float(magnetization_m),
        absorption=float(absorption),
        frustration=float(frustration),
        below_at_line=below_at_line,
        resilience=resilience,
        shock_response=shock_response,
        description=description,
    )


def _compute_frustration(corr_matrix: np.ndarray) -> float:
    """
    Compute frustration ratio from correlation matrix.

    Frustrated triangle: sign(ρ₁₂) × sign(ρ₂₃) × sign(ρ₁₃) < 0

    High frustration = complex energy landscape = many metastable states
    """
    n = corr_matrix.shape[0]
    if n < 3:
        return 0.0

    n_triangles = 0
    n_frustrated = 0

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Triangle (i, j, k)
                product = np.sign(corr_matrix[i, j]) * \
                          np.sign(corr_matrix[j, k]) * \
                          np.sign(corr_matrix[i, k])

                n_triangles += 1
                if product < 0:
                    n_frustrated += 1

    return n_frustrated / n_triangles if n_triangles > 0 else 0.0


def _empty_spin_glass_result() -> SpinGlassResult:
    """Return empty spin glass result."""
    return SpinGlassResult(
        phase=SpinGlassPhase.UNKNOWN,
        overlap_q=0.0,
        magnetization_m=0.0,
        absorption=1.0,
        frustration=0.0,
        below_at_line=False,
        resilience="UNKNOWN",
        shock_response="unknown",
        description="Insufficient data for spin glass analysis",
    )


def interpret_spin_glass(result: SpinGlassResult, domain: str) -> str:
    """
    Provide domain-specific interpretation of spin glass phase.

    Args:
        result: SpinGlassResult from compute_spin_glass
        domain: Domain identifier

    Returns:
        Domain-specific interpretation
    """
    phase = result.phase

    if domain in ["markets", "finance", "trading"]:
        if phase == SpinGlassPhase.PARAMAGNETIC:
            return "Market regime: Normal conditions. Assets uncorrelated. Diversification works."
        elif phase == SpinGlassPhase.FERROMAGNETIC:
            return "Market regime: Trending market. Assets moving together. Momentum strategies favored."
        elif phase == SpinGlassPhase.SPIN_GLASS:
            return "Market regime: FRAGILE STATE. High correlation but no direction. Volatility spike imminent."
        elif phase == SpinGlassPhase.MIXED:
            return "Market regime: CRITICAL. Trend with uncertainty. Major transition possible."

    elif domain in ["turbofan", "engine", "industrial"]:
        if phase == SpinGlassPhase.PARAMAGNETIC:
            return "Equipment health: NOMINAL. Sensors independent. Normal operation."
        elif phase == SpinGlassPhase.FERROMAGNETIC:
            return "Equipment health: DEGRADING. Sensors trending together. Performance declining."
        elif phase == SpinGlassPhase.SPIN_GLASS:
            return "Equipment health: CRITICAL. Multiple failure modes possible. Immediate attention."
        elif phase == SpinGlassPhase.MIXED:
            return "Equipment health: UNSTABLE. Trending toward failure with uncertainty."

    elif domain in ["building", "structure", "bridge"]:
        if phase == SpinGlassPhase.PARAMAGNETIC:
            return "Structural health: STABLE. Vibration modes independent. Normal operation."
        elif phase == SpinGlassPhase.FERROMAGNETIC:
            return "Structural health: ALERT. Modes aligning. Check for resonance conditions."
        elif phase == SpinGlassPhase.SPIN_GLASS:
            return "Structural health: WARNING. Complex mode interactions. Monitor carefully."
        elif phase == SpinGlassPhase.MIXED:
            return "Structural health: CRITICAL. Mode coupling with regime change. Structural assessment needed."

    return result.description


def generate_spin_glass_report(
    result: SpinGlassResult,
    domain: str = "general",
) -> str:
    """
    Generate human-readable spin glass report.

    Args:
        result: SpinGlassResult
        domain: Domain for interpretation

    Returns:
        Formatted report string
    """
    report = f"""
SPIN GLASS INTERPRETATION
═══════════════════════════════════════════════════════════════════════

ORDER PARAMETERS
├── Overlap q:        {result.overlap_q:.3f}  (0=independent, 1=locked)
├── Magnetization m:  {result.magnetization_m:.3f}  (net trend direction)
├── Absorption:       {result.absorption:.1%}  (resilience capacity)
└── Frustration:      {result.frustration:.3f}  (complexity)

PHASE CLASSIFICATION
├── Phase:            {result.phase.value.upper()}
├── Below AT line:    {"YES - complex landscape" if result.below_at_line else "NO - simple landscape"}
├── Resilience:       {result.resilience}
└── Shock response:   {result.shock_response}

INTERPRETATION
{result.description}

DOMAIN CONTEXT ({domain})
{interpret_spin_glass(result, domain)}
═══════════════════════════════════════════════════════════════════════
"""
    return report
