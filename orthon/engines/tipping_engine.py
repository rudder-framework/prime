"""
Tipping Classification Engine

Identify failure mechanism based on causality between geometry and mass.

Types:
- B-tipping: Bifurcation-induced (Geometry → Mass) - CSD provides warning
- R-tipping: Rate-induced (Mass → Geometry) - NO early warning
- Resonance: Bidirectional coupling - Coupling increase is warning
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List


class TippingType(Enum):
    """Tipping point classification."""
    B_TIPPING = "b_tipping"     # Bifurcation-induced (CSD warning)
    R_TIPPING = "r_tipping"     # Rate-induced (no warning)
    RESONANCE = "resonance"     # Bidirectional feedback
    NONE = "none"               # No tipping detected


@dataclass
class TippingResult:
    """Result of tipping classification."""
    tipping_type: TippingType
    causality_direction: str        # "geometry→mass", "mass→geometry", "bidirectional"
    granger_g_to_m: float          # Granger F-stat (geometry causes mass)
    granger_m_to_g: float          # Granger F-stat (mass causes geometry)
    warning_available: bool         # Is early warning possible?
    warning_mechanism: str         # What to watch
    confidence: float
    description: str


def classify_tipping(
    geometry_series: np.ndarray,
    mass_series: np.ndarray,
    csd_detected: bool = False,
    max_lag: int = 5,
) -> TippingResult:
    """
    Classify tipping type based on causality analysis.

    Args:
        geometry_series: Time series of eff_dim
        mass_series: Time series of total variance
        csd_detected: Whether CSD has been detected
        max_lag: Maximum lag for Granger causality

    Returns:
        TippingResult with classification and causality metrics
    """
    n = len(geometry_series)
    if n < 20 or n != len(mass_series):
        return TippingResult(
            tipping_type=TippingType.NONE,
            causality_direction="unknown",
            granger_g_to_m=0.0,
            granger_m_to_g=0.0,
            warning_available=False,
            warning_mechanism="Insufficient data",
            confidence=0.0,
            description="Insufficient data for tipping classification",
        )

    # Compute Granger causality
    g_to_m, g_to_m_p = _granger_causality(geometry_series, mass_series, max_lag)
    m_to_g, m_to_g_p = _granger_causality(mass_series, geometry_series, max_lag)

    # Significance threshold
    alpha = 0.05

    g_causes_m = g_to_m_p < alpha
    m_causes_g = m_to_g_p < alpha

    # Classify tipping type
    if g_causes_m and not m_causes_g:
        # Geometry → Mass: B-tipping
        if csd_detected:
            tipping_type = TippingType.B_TIPPING
            warning = True
            warning_mechanism = "CSD provides early warning (AR1↑, Var↑)"
            description = "B-tipping: Geometry collapse drives mass change. CSD detected."
        else:
            tipping_type = TippingType.B_TIPPING
            warning = True
            warning_mechanism = "Monitor CSD indicators (AR1, variance trends)"
            description = "B-tipping pattern: Geometry leads mass. Watch for CSD."
        causality = "geometry→mass"
        confidence = 1 - g_to_m_p

    elif m_causes_g and not g_causes_m:
        # Mass → Geometry: R-tipping
        tipping_type = TippingType.R_TIPPING
        warning = False
        warning_mechanism = "NO EARLY WARNING - rate too fast"
        causality = "mass→geometry"
        confidence = 1 - m_to_g_p
        description = "R-tipping: Mass change drives geometry collapse. No CSD warning."

    elif g_causes_m and m_causes_g:
        # Bidirectional: Resonance
        tipping_type = TippingType.RESONANCE
        warning = True
        warning_mechanism = "Coupling strength increase is warning signal"
        causality = "bidirectional"
        confidence = (1 - g_to_m_p + 1 - m_to_g_p) / 2
        description = "Resonance: Bidirectional feedback between geometry and mass."

    else:
        # No causal relationship
        tipping_type = TippingType.NONE
        warning = False
        warning_mechanism = "No significant causality detected"
        causality = "none"
        confidence = 0.5
        description = "No tipping pattern detected in current data."

    return TippingResult(
        tipping_type=tipping_type,
        causality_direction=causality,
        granger_g_to_m=float(g_to_m),
        granger_m_to_g=float(m_to_g),
        warning_available=warning,
        warning_mechanism=warning_mechanism,
        confidence=float(confidence),
        description=description,
    )


def _granger_causality(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[float, float]:
    """
    Test if x Granger-causes y.

    Restricted:   yₜ = α + Σᵢ βᵢyₜ₋ᵢ + εₜ
    Unrestricted: yₜ = α + Σᵢ βᵢyₜ₋ᵢ + Σⱼ γⱼxₜ₋ⱼ + εₜ

    F-test: Does adding x improve prediction of y?

    Returns:
        Tuple of (F_statistic, p_value)
    """
    n = len(x)
    if n <= 2 * max_lag + 1:
        return 0.0, 1.0

    # Create lagged matrices
    Y = y[max_lag:]
    n_obs = len(Y)

    # Lagged Y (for both models)
    Y_lags = np.column_stack([y[max_lag - i - 1:n - i - 1] for i in range(max_lag)])

    # Lagged X (for unrestricted model)
    X_lags = np.column_stack([x[max_lag - i - 1:n - i - 1] for i in range(max_lag)])

    # Add constant
    ones = np.ones((n_obs, 1))

    # Restricted model: Y ~ Y_lags
    Z_restricted = np.hstack([ones, Y_lags])

    # Unrestricted model: Y ~ Y_lags + X_lags
    Z_unrestricted = np.hstack([ones, Y_lags, X_lags])

    try:
        # Fit restricted model
        beta_r, residuals_r, _, _ = np.linalg.lstsq(Z_restricted, Y, rcond=None)
        if len(residuals_r) == 0:
            ssr_r = np.sum((Y - Z_restricted @ beta_r) ** 2)
        else:
            ssr_r = residuals_r[0]

        # Fit unrestricted model
        beta_u, residuals_u, _, _ = np.linalg.lstsq(Z_unrestricted, Y, rcond=None)
        if len(residuals_u) == 0:
            ssr_u = np.sum((Y - Z_unrestricted @ beta_u) ** 2)
        else:
            ssr_u = residuals_u[0]

        # F-test
        df1 = max_lag  # Additional parameters
        df2 = n_obs - 2 * max_lag - 1  # Residual DF

        if df2 <= 0 or ssr_u < 1e-10:
            return 0.0, 1.0

        F_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)

        # P-value from F distribution
        from scipy.stats import f
        p_value = 1 - f.cdf(F_stat, df1, df2)

        return float(F_stat), float(p_value)

    except Exception:
        return 0.0, 1.0


def interpret_tipping_for_domain(
    tipping: TippingResult,
    domain: str,
) -> str:
    """
    Provide domain-specific interpretation of tipping classification.

    Args:
        tipping: TippingResult from classify_tipping
        domain: Domain identifier

    Returns:
        Domain-specific interpretation
    """
    t = tipping.tipping_type

    if domain in ["markets", "finance"]:
        if t == TippingType.B_TIPPING:
            return "Market stress building: Correlation surge precedes volatility spike. CSD warning available."
        elif t == TippingType.R_TIPPING:
            return "Flash event risk: Volatility can trigger correlation cascade with no warning."
        elif t == TippingType.RESONANCE:
            return "Feedback loop: Correlation and volatility reinforcing. Monitor coupling."

    elif domain in ["turbofan", "engine", "industrial"]:
        if t == TippingType.R_TIPPING:
            return "DEGRADATION WARNING: Component failure driving geometry collapse. No CSD warning available."
        elif t == TippingType.B_TIPPING:
            return "Structural degradation: Geometry change precedes performance loss."
        elif t == TippingType.RESONANCE:
            return "Resonance condition: Vibration modes coupling with performance."

    elif domain in ["building", "structure"]:
        if t == TippingType.RESONANCE:
            return "RESONANCE DANGER: Structural modes coupling with excitation. Critical monitoring required."
        elif t == TippingType.B_TIPPING:
            return "Structural fatigue: Mode shape changes preceding damage."

    return tipping.description
