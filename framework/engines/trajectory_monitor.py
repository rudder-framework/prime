"""
Trajectory Monitor Engine (RUDDER).

Interprets PRISM trajectory sensitivity outputs to provide:
- Real-time sensitivity alerts
- Saddle proximity warnings
- Variable importance recommendations
- Regime transition detection

PRISM computes, RUDDER interprets and acts.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class TrajectoryAlert(Enum):
    """Alert levels for trajectory monitoring."""
    NORMAL = "normal"
    ATTENTION = "attention"        # Elevated sensitivity
    WARNING = "warning"            # Approaching saddle
    CRITICAL = "critical"          # Near saddle, possible transition


class SensitivityChange(Enum):
    """Types of sensitivity transitions."""
    STABLE = "stable"              # Same dominant variable
    SHIFT = "shift"                # Smooth transition
    TRANSITION = "transition"      # Abrupt change


@dataclass
class TrajectoryStatus:
    """Current trajectory status assessment."""
    alert_level: TrajectoryAlert
    dominant_variable: Optional[str]
    sensitivity_ranking: List[Tuple[str, float]]
    saddle_proximity: float
    basin_stability: float
    recommendations: List[str]
    message: str


@dataclass
class SensitivityReport:
    """Sensitivity analysis report."""
    current_dominant: str
    sensitivity_by_variable: Dict[str, float]
    sensitivity_trend: str  # increasing, decreasing, stable
    recent_transitions: List[Dict]
    focus_variables: List[str]  # Variables to monitor closely


def interpret_ftle(
    ftle_mean: Optional[float],
    ftle_current: Optional[float],
    ftle_std: Optional[float],
) -> Tuple[str, str]:
    """
    Interpret FTLE values.

    Returns:
        (regime, description)
    """
    if ftle_current is None:
        return "unknown", "Insufficient data for FTLE estimation"

    if ftle_current > 0.5:
        regime = "highly_sensitive"
        desc = f"FTLE={ftle_current:.3f}: Strong exponential divergence. Trajectory is in highly sensitive region."
    elif ftle_current > 0.1:
        regime = "sensitive"
        desc = f"FTLE={ftle_current:.3f}: Moderate sensitivity. Perturbations amplify over time."
    elif ftle_current > 0.01:
        regime = "weakly_sensitive"
        desc = f"FTLE={ftle_current:.3f}: Weak sensitivity. Near neutral stability."
    elif ftle_current > -0.01:
        regime = "neutral"
        desc = f"FTLE={ftle_current:.3f}: Neutral stability. Perturbations neither grow nor decay."
    elif ftle_current > -0.1:
        regime = "weakly_stable"
        desc = f"FTLE={ftle_current:.3f}: Weakly stable. Perturbations decay slowly."
    else:
        regime = "stable"
        desc = f"FTLE={ftle_current:.3f}: Stable. Perturbations decay exponentially."

    return regime, desc


def interpret_saddle_proximity(
    saddle_score: Optional[float],
    basin_stability: Optional[float],
    stability_type: str,
) -> Tuple[TrajectoryAlert, str]:
    """
    Interpret saddle proximity and basin stability.

    Returns:
        (alert_level, message)
    """
    if saddle_score is None:
        return TrajectoryAlert.NORMAL, "Saddle detection unavailable"

    if stability_type == "saddle":
        if saddle_score > 0.8:
            return TrajectoryAlert.CRITICAL, "At saddle point! System at unstable equilibrium."
        elif saddle_score > 0.5:
            return TrajectoryAlert.WARNING, f"Near saddle (score={saddle_score:.2f}). Potential basin transition."

    if basin_stability is not None:
        if basin_stability < 0.2:
            return TrajectoryAlert.WARNING, f"Low basin stability ({basin_stability:.2f}). System may be near basin boundary."
        elif basin_stability < 0.5:
            return TrajectoryAlert.ATTENTION, f"Moderate basin stability ({basin_stability:.2f}). Monitor for transitions."

    if saddle_score > 0.3:
        return TrajectoryAlert.ATTENTION, f"Elevated saddle proximity (score={saddle_score:.2f})"

    return TrajectoryAlert.NORMAL, f"Stable trajectory (saddle_score={saddle_score:.2f})"


def generate_sensitivity_report(
    sensitivity_result: Dict[str, Any],
    window: int = 50,
) -> SensitivityReport:
    """
    Generate sensitivity analysis report from PRISM output.

    Args:
        sensitivity_result: Output from trajectory_sensitivity engine
        window: Window for trend analysis

    Returns:
        SensitivityReport with interpretation
    """
    signal_sensitivity = sensitivity_result.get('signal_sensitivity', {})
    dominant = sensitivity_result.get('dominant_variable', 'unknown')
    transitions = sensitivity_result.get('transitions', [])

    # Build sensitivity dict
    sens_by_var = {}
    for name, info in signal_sensitivity.items():
        sens = info.get('mean_sensitivity')
        if sens is not None and not np.isnan(sens):
            sens_by_var[name] = sens

    # Determine trend from sensitivity entropy
    entropy = sensitivity_result.get('sensitivity_entropy')
    if entropy is not None and len(entropy) > window:
        recent = entropy[-window:]
        valid = ~np.isnan(recent)
        if np.sum(valid) > 10:
            early = np.nanmean(recent[:len(recent)//2])
            late = np.nanmean(recent[len(recent)//2:])
            if late > early + 0.1:
                trend = "diffusing"  # Sensitivity spreading across variables
            elif late < early - 0.1:
                trend = "focusing"  # Sensitivity concentrating on fewer variables
            else:
                trend = "stable"
        else:
            trend = "unknown"
    else:
        trend = "unknown"

    # Recent transitions
    recent_transitions = transitions[-5:] if transitions else []

    # Focus variables (top 3 by mean sensitivity)
    sorted_vars = sorted(sens_by_var.items(), key=lambda x: -x[1])
    focus_vars = [name for name, _ in sorted_vars[:3]]

    return SensitivityReport(
        current_dominant=dominant if dominant else "unknown",
        sensitivity_by_variable=sens_by_var,
        sensitivity_trend=trend,
        recent_transitions=recent_transitions,
        focus_variables=focus_vars,
    )


def generate_recommendations(
    ftle_result: Dict[str, Any],
    saddle_result: Dict[str, Any],
    sensitivity_result: Dict[str, Any],
) -> List[str]:
    """
    Generate actionable recommendations based on trajectory analysis.

    Args:
        ftle_result: FTLE engine output
        saddle_result: Saddle detection output
        sensitivity_result: Sensitivity analysis output

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # FTLE-based recommendations
    ftle_current = ftle_result.get('ftle_current')
    if ftle_current is not None:
        if ftle_current > 0.3:
            recommendations.append(
                "HIGH SENSITIVITY: System in sensitive regime. Small perturbations "
                "will amplify. Consider reducing control gains or operating more conservatively."
            )
        elif ftle_current < -0.1:
            recommendations.append(
                "STABLE REGIME: System is in stable region. Good opportunity for "
                "setpoint changes or maintenance operations."
            )

    # Saddle-based recommendations
    saddle_score = saddle_result.get('saddle_score_current')
    if saddle_score is not None and saddle_score > 0.5:
        stability_type = saddle_result.get('current_stability_type', 'unknown')
        recommendations.append(
            f"SADDLE PROXIMITY ({stability_type}): System near unstable equilibrium. "
            "Avoid aggressive control actions. Monitor for regime transition."
        )

    basin_stability = saddle_result.get('basin_stability_current')
    if basin_stability is not None and basin_stability < 0.3:
        recommendations.append(
            "LOW BASIN STABILITY: System may transition to different operating mode. "
            "Review control setpoints and consider stabilizing actions."
        )

    # Sensitivity-based recommendations
    dominant = sensitivity_result.get('dominant_variable')
    if dominant:
        recommendations.append(
            f"FOCUS VARIABLE: '{dominant}' currently has highest sensitivity. "
            f"Prioritize monitoring this variable."
        )

    n_transitions = sensitivity_result.get('n_transitions', 0)
    if n_transitions > 5:
        recommendations.append(
            f"FREQUENT TRANSITIONS: {n_transitions} sensitivity transitions detected. "
            "System may be operating in transitional regime."
        )

    if not recommendations:
        recommendations.append("System operating normally. Continue standard monitoring.")

    return recommendations


def assess_trajectory_status(
    ftle_result: Dict[str, Any],
    saddle_result: Dict[str, Any],
    sensitivity_result: Dict[str, Any],
) -> TrajectoryStatus:
    """
    Comprehensive trajectory status assessment.

    Args:
        ftle_result: FTLE engine output
        saddle_result: Saddle detection output
        sensitivity_result: Sensitivity analysis output

    Returns:
        TrajectoryStatus with full assessment
    """
    # FTLE interpretation
    ftle_regime, ftle_desc = interpret_ftle(
        ftle_result.get('ftle_mean'),
        ftle_result.get('ftle_current'),
        ftle_result.get('ftle_std'),
    )

    # Saddle interpretation
    saddle_alert, saddle_msg = interpret_saddle_proximity(
        saddle_result.get('saddle_score_current'),
        saddle_result.get('basin_stability_current'),
        saddle_result.get('current_stability_type', 'unknown'),
    )

    # Sensitivity report
    sens_report = generate_sensitivity_report(sensitivity_result)

    # Determine overall alert level
    ftle_current = ftle_result.get('ftle_current')
    if saddle_alert == TrajectoryAlert.CRITICAL:
        alert_level = TrajectoryAlert.CRITICAL
    elif saddle_alert == TrajectoryAlert.WARNING:
        alert_level = TrajectoryAlert.WARNING
    elif ftle_current is not None and ftle_current > 0.3:
        alert_level = TrajectoryAlert.WARNING
    elif saddle_alert == TrajectoryAlert.ATTENTION or (ftle_current is not None and ftle_current > 0.1):
        alert_level = TrajectoryAlert.ATTENTION
    else:
        alert_level = TrajectoryAlert.NORMAL

    # Build sensitivity ranking
    ranking = sorted(
        sens_report.sensitivity_by_variable.items(),
        key=lambda x: -x[1]
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        ftle_result, saddle_result, sensitivity_result
    )

    # Build summary message
    messages = [ftle_desc, saddle_msg]
    if sens_report.current_dominant:
        messages.append(f"Dominant variable: {sens_report.current_dominant}")
    message = " | ".join(messages)

    return TrajectoryStatus(
        alert_level=alert_level,
        dominant_variable=sens_report.current_dominant,
        sensitivity_ranking=ranking,
        saddle_proximity=saddle_result.get('saddle_score_current', 0.0) or 0.0,
        basin_stability=saddle_result.get('basin_stability_current', 1.0) or 1.0,
        recommendations=recommendations,
        message=message,
    )


def monitor_trajectory(
    ftle_result: Dict[str, Any],
    saddle_result: Dict[str, Any],
    sensitivity_result: Dict[str, Any],
    previous_status: Optional[TrajectoryStatus] = None,
) -> Tuple[TrajectoryStatus, List[str]]:
    """
    Main monitoring function for real-time trajectory analysis.

    Args:
        ftle_result: Current FTLE engine output
        saddle_result: Current saddle detection output
        sensitivity_result: Current sensitivity analysis output
        previous_status: Previous status for change detection

    Returns:
        (current_status, alerts): Current status and any new alerts
    """
    status = assess_trajectory_status(ftle_result, saddle_result, sensitivity_result)

    alerts = []

    # Check for status changes
    if previous_status is not None:
        # Alert level change
        if status.alert_level != previous_status.alert_level:
            if status.alert_level.value in ['warning', 'critical']:
                alerts.append(
                    f"ALERT LEVEL CHANGE: {previous_status.alert_level.value} -> {status.alert_level.value}"
                )

        # Dominant variable change
        if (status.dominant_variable != previous_status.dominant_variable
                and status.dominant_variable is not None):
            alerts.append(
                f"SENSITIVITY SHIFT: Dominant variable changed from "
                f"'{previous_status.dominant_variable}' to '{status.dominant_variable}'"
            )

        # Basin stability drop
        if (previous_status.basin_stability > 0.5 and status.basin_stability < 0.3):
            alerts.append(
                f"STABILITY DROP: Basin stability decreased from "
                f"{previous_status.basin_stability:.2f} to {status.basin_stability:.2f}"
            )

    # Critical conditions always alert
    if status.alert_level == TrajectoryAlert.CRITICAL:
        alerts.append(f"CRITICAL: {status.message}")

    return status, alerts
