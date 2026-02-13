"""
Dynamics Interpreter
====================

Interprets dynamical systems metrics from PRISM for stability analysis.

Complements PhysicsInterpreter with:
- Lyapunov exponent analysis (chaos detection)
- Basin stability scoring
- Regime transition detection
- Birth certificate (early-life prognosis)

The key insight: failure is loss of dynamical stability.
We measure how stable your system is.
"""

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple


@dataclass
class StabilityDiagnosis:
    """Diagnosis of dynamical stability state."""
    stability_class: str  # 'STABLE', 'MARGINAL', 'UNSTABLE', 'CHAOTIC'
    basin_score: float  # 0-1, higher = more stable
    lyapunov_max: Optional[float]  # Maximum Lyapunov exponent
    regime: str  # 'STABLE', 'DESTABILIZING', 'TRANSIENT'
    birth_grade: Optional[str]  # 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
    prognosis: str  # Human-readable prognosis


class DynamicsInterpreter:
    """
    Interprets dynamical systems metrics from PRISM.

    Focuses on stability analysis:
    - How stable is the current operating state?
    - Is the system approaching a regime change?
    - What does early-life behavior predict about lifespan?
    """

    def __init__(
        self,
        prism_output: Optional[str] = None,
        physics_path: Optional[str] = None,
        physics_df: Optional[pl.DataFrame] = None,
        dynamics_path: Optional[str] = None,
        dynamics_df: Optional[pl.DataFrame] = None,
        primitives_path: Optional[str] = None,
        primitives_df: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize interpreter.

        Args:
            prism_output: Path to PRISM output directory (contains physics.parquet, dynamics.parquet)
            physics_path: Path to physics.parquet (alternative to prism_output)
            physics_df: Or provide DataFrame directly
            dynamics_path: Path to dynamics.parquet (window-level Lyapunov + RQA)
            dynamics_df: Or provide DataFrame directly
            primitives_path: Path to primitives.parquet (signal-level, legacy)
            primitives_df: Or provide DataFrame directly
        """
        # Resolve paths from prism_output directory
        if prism_output:
            output_dir = Path(prism_output)
            if physics_path is None:
                physics_path = output_dir / "physics.parquet"
            if dynamics_path is None:
                dynamics_path = output_dir / "dynamics.parquet"
            if primitives_path is None:
                primitives_path = output_dir / "primitives.parquet"

        # Convert string paths to Path objects
        if physics_path and not isinstance(physics_path, Path):
            physics_path = Path(physics_path)
        if dynamics_path and not isinstance(dynamics_path, Path):
            dynamics_path = Path(dynamics_path)
        if primitives_path and not isinstance(primitives_path, Path):
            primitives_path = Path(primitives_path)

        # Load physics
        if physics_df is not None:
            self.physics = physics_df
        elif physics_path and physics_path.exists():
            self.physics = pl.read_parquet(physics_path)
        else:
            self.physics = None

        # Load dynamics (window-level Lyapunov + RQA) - preferred source
        if dynamics_df is not None:
            self.dynamics = dynamics_df
        elif dynamics_path and dynamics_path.exists():
            self.dynamics = pl.read_parquet(dynamics_path)
        else:
            self.dynamics = None

        # Load primitives (signal-level, legacy fallback)
        if primitives_df is not None:
            self.primitives = primitives_df
        elif primitives_path and primitives_path.exists():
            self.primitives = pl.read_parquet(primitives_path)
        else:
            self.primitives = None

        self._entity_cache: Dict[str, Dict] = {}

    def get_entities(self) -> List[str]:
        """Get list of entity IDs."""
        if self.physics is not None:
            return self.physics["entity_id"].unique().sort().to_list()
        return []

    def analyze_lyapunov(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze Lyapunov exponents and RQA metrics for an entity.

        Uses dynamics.parquet (window-level) as primary source,
        falls back to primitives.parquet (signal-level) if needed.

        Lyapunov exponent measures trajectory divergence:
        - λ > 0.1: Chaotic (trajectories diverge exponentially)
        - λ > 0.02: Weakly unstable
        - λ > 0: Marginal stability
        - λ < 0: Stable (trajectories converge)
        """
        # Try dynamics.parquet first (window-level with RQA)
        if self.dynamics is not None:
            entity_data = self.dynamics.filter(pl.col("entity_id") == entity_id)
            if not entity_data.is_empty() and "lyapunov_max" in entity_data.columns:
                lyapunov = entity_data["lyapunov_max"].drop_nulls().to_numpy()
                if len(lyapunov) > 0:
                    lyap_max = float(np.max(lyapunov))
                    lyap_mean = float(np.mean(lyapunov))
                    n_chaotic = int(np.sum(lyapunov > 0.1))
                    n_unstable = int(np.sum(lyapunov > 0.02))

                    # RQA metrics
                    det = entity_data["determinism"].drop_nulls().to_numpy()
                    lam = entity_data["laminarity"].drop_nulls().to_numpy()
                    rqa_ent = entity_data["rqa_entropy"].drop_nulls().to_numpy()
                    trap = entity_data["trapping_time"].drop_nulls().to_numpy()

                    # Classification
                    if lyap_max > 0.1:
                        stability_class = "CHAOTIC"
                    elif lyap_mean > 0.02 and np.mean(det) < 0.1:
                        stability_class = "UNSTABLE"
                    elif lyap_mean > 0:
                        stability_class = "WEAKLY_UNSTABLE"
                    elif lyap_mean > -0.02:
                        stability_class = "MARGINAL"
                    else:
                        stability_class = "STABLE"

                    # Dynamics type from RQA
                    mean_det = float(np.mean(det)) if len(det) > 0 else 0
                    if mean_det > 0.5:
                        dynamics_type = "DETERMINISTIC"
                    elif mean_det > 0.2:
                        dynamics_type = "MIXED"
                    else:
                        dynamics_type = "STOCHASTIC"

                    return {
                        "lyapunov_max": lyap_max,
                        "lyapunov_mean": round(lyap_mean, 4),
                        "n_chaotic_windows": n_chaotic,
                        "n_unstable_windows": n_unstable,
                        "n_windows": len(lyapunov),
                        "stability_class": stability_class,
                        "dynamics_type": dynamics_type,
                        "determinism": round(mean_det, 3),
                        "laminarity": round(float(np.mean(lam)), 3) if len(lam) > 0 else None,
                        "rqa_entropy": round(float(np.mean(rqa_ent)), 3) if len(rqa_ent) > 0 else None,
                        "trapping_time": round(float(np.mean(trap)), 2) if len(trap) > 0 else None,
                    }

        # Fallback to primitives.parquet (signal-level, legacy)
        if self.primitives is not None:
            entity_data = self.primitives.filter(pl.col("entity_id") == entity_id)
            if not entity_data.is_empty():
                # Try both column names
                lyap_col = "lyapunov" if "lyapunov" in entity_data.columns else "lyapunov_exponent"
                if lyap_col in entity_data.columns:
                    lyapunov = entity_data[lyap_col].drop_nulls().to_numpy()
                    if len(lyapunov) > 0:
                        lyap_max = float(np.max(lyapunov))
                        lyap_mean = float(np.mean(lyapunov))
                        n_chaotic = int(np.sum(lyapunov > 0.1))

                        if lyap_max > 0.1:
                            stability_class = "CHAOTIC"
                        elif lyap_max > 0:
                            stability_class = "UNSTABLE"
                        elif lyap_max > -0.1:
                            stability_class = "MARGINAL"
                        else:
                            stability_class = "STABLE"

                        return {
                            "lyapunov_max": lyap_max,
                            "lyapunov_mean": round(lyap_mean, 4),
                            "n_chaotic_signals": n_chaotic,
                            "n_signals": len(lyapunov),
                            "stability_class": stability_class,
                        }

        return None

    def compute_basin_stability(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Compute basin stability score.

        Basin stability measures how "deep" the attractor basin is:
        - Deep basin: tolerates large perturbations
        - Shallow basin: vulnerable to transitions
        """
        if self.physics is None:
            return None

        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)
        if entity_data.is_empty():
            return None

        # Extract metrics
        coherence = entity_data["coherence"].to_numpy()
        velocity = entity_data["state_velocity"].to_numpy() if "state_velocity" in entity_data.columns else np.zeros(len(coherence))

        # Compute components
        mean_coherence = float(np.mean(coherence))
        coherence_volatility = float(np.std(coherence))
        mean_velocity = float(np.nanmean(velocity))
        velocity_volatility = float(np.nanstd(velocity))

        # Fleet-relative scoring would need fleet stats
        # For single entity, use absolute thresholds
        coherence_score = 1.0 / (1.0 + np.exp(-10 * (mean_coherence - 0.5)))
        velocity_score = 1.0 / (1.0 + np.exp(10 * (mean_velocity - 0.3)))
        coh_stability = 1.0 / (1.0 + np.exp(20 * (coherence_volatility - 0.1)))
        vel_stability = 1.0 / (1.0 + np.exp(5 * (velocity_volatility - 0.5)))

        basin_score = (
            coherence_score * 0.30 +
            velocity_score * 0.30 +
            coh_stability * 0.20 +
            vel_stability * 0.20
        )

        # Classification
        if basin_score > 0.7:
            basin_class = "DEEP_BASIN"
        elif basin_score > 0.5:
            basin_class = "MODERATE_BASIN"
        elif basin_score > 0.3:
            basin_class = "SHALLOW_BASIN"
        else:
            basin_class = "UNSTABLE"

        return {
            "basin_stability_score": round(basin_score, 3),
            "basin_class": basin_class,
            "mean_coherence": round(mean_coherence, 3),
            "mean_velocity": round(mean_velocity, 4),
            "coherence_volatility": round(coherence_volatility, 4),
            "velocity_volatility": round(velocity_volatility, 4),
        }

    def detect_regime(self, entity_id: str, window: int = 20) -> Optional[Dict[str, Any]]:
        """
        Detect current regime and recent transitions.

        Regimes:
        - STABLE: Low velocity, consistent coherence
        - DESTABILIZING: Increasing velocity, decreasing coherence
        - TRANSIENT: High volatility, regime uncertainty
        """
        if self.physics is None:
            return None

        entity_data = self.physics.filter(pl.col("entity_id") == entity_id).sort("I")
        if len(entity_data) < window:
            return None

        # Get recent window
        coherence = entity_data["coherence"].to_numpy()
        velocity = entity_data["state_velocity"].to_numpy() if "state_velocity" in entity_data.columns else np.zeros(len(coherence))

        recent_coh = coherence[-window:]
        recent_vel = velocity[-window:]

        # Compute trends
        if len(recent_coh) >= 10:
            x = np.arange(len(recent_coh))
            coh_trend = np.polyfit(x, recent_coh, 1)[0]
            vel_trend = np.polyfit(x, recent_vel, 1)[0] if not np.any(np.isnan(recent_vel)) else 0
        else:
            coh_trend = 0
            vel_trend = 0

        # Recent volatility
        coh_volatility = float(np.std(recent_coh))
        vel_volatility = float(np.nanstd(recent_vel))

        # Determine regime
        if coh_trend < -0.005 and vel_trend > 0.005:
            regime = "DESTABILIZING"
        elif coh_trend > 0.005 and vel_trend < -0.005:
            regime = "STABILIZING"
        elif coh_volatility > 0.1 or vel_volatility > 0.5:
            regime = "TRANSIENT"
        else:
            regime = "STABLE"

        return {
            "current_regime": regime,
            "coherence_trend": round(coh_trend, 5),
            "velocity_trend": round(vel_trend, 5),
            "coherence_volatility": round(coh_volatility, 4),
            "velocity_volatility": round(vel_volatility, 4),
            "recent_coherence": round(float(np.mean(recent_coh)), 3),
            "recent_velocity": round(float(np.nanmean(recent_vel)), 4),
        }

    def compute_birth_certificate(self, entity_id: str, early_pct: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        Compute birth certificate from early-life metrics.

        Birth certificate captures early-life stability which
        correlates with total lifespan in run-to-failure data.
        """
        if self.physics is None:
            return None

        entity_data = self.physics.filter(pl.col("entity_id") == entity_id).sort("I")
        n = len(entity_data)
        if n < 20:
            return None

        early_n = max(10, int(n * early_pct))

        # Get early-life data
        early_data = entity_data.head(early_n)

        early_coherence = early_data["coherence"].to_numpy()
        early_velocity = early_data["state_velocity"].to_numpy() if "state_velocity" in early_data.columns else np.zeros(early_n)

        # Early-life metrics
        mean_coh = float(np.mean(early_coherence))
        std_coh = float(np.std(early_coherence))
        mean_vel = float(np.nanmean(early_velocity))

        # Birth certificate components (use absolute thresholds)
        coupling_score = 1.0 / (1.0 + np.exp(-10 * (mean_coh - 0.5)))
        stability_score = 1.0 / (1.0 + np.exp(10 * (mean_vel - 0.2)))
        consistency_score = 1.0 / (1.0 + np.exp(20 * (std_coh - 0.05)))

        birth_score = (
            coupling_score * 0.4 +
            stability_score * 0.4 +
            consistency_score * 0.2
        )

        # Grade
        if birth_score > 0.65:
            grade = "EXCELLENT"
            prognosis = "Strong early-life metrics. Expected long operational life."
        elif birth_score > 0.5:
            grade = "GOOD"
            prognosis = "Healthy start. Above-average lifespan expected."
        elif birth_score > 0.35:
            grade = "FAIR"
            prognosis = "Average early-life metrics. Monitor for early degradation."
        else:
            grade = "POOR"
            prognosis = "Weak early-life metrics. High risk of early failure."

        return {
            "birth_certificate_score": round(birth_score, 3),
            "birth_grade": grade,
            "prognosis": prognosis,
            "early_coherence": round(mean_coh, 3),
            "early_velocity": round(mean_vel, 4),
            "early_volatility": round(std_coh, 4),
            "observations_used": early_n,
            "total_observations": n,
        }

    def analyze_stability(self, entity_id: str) -> StabilityDiagnosis:
        """
        Full stability analysis for an entity.

        Combines:
        - Lyapunov analysis (if available)
        - Basin stability
        - Regime detection
        - Birth certificate
        """
        # Get components
        lyapunov = self.analyze_lyapunov(entity_id)
        basin = self.compute_basin_stability(entity_id)
        regime = self.detect_regime(entity_id)
        birth = self.compute_birth_certificate(entity_id)

        # Determine overall stability class
        if lyapunov and lyapunov["stability_class"] == "CHAOTIC":
            stability_class = "CHAOTIC"
        elif basin and basin["basin_class"] == "UNSTABLE":
            stability_class = "UNSTABLE"
        elif lyapunov and lyapunov["stability_class"] == "UNSTABLE":
            stability_class = "UNSTABLE"
        elif basin and basin["basin_class"] == "SHALLOW_BASIN":
            stability_class = "MARGINAL"
        elif lyapunov and lyapunov["stability_class"] == "MARGINAL":
            stability_class = "MARGINAL"
        else:
            stability_class = "STABLE"

        # Basin score
        basin_score = basin["basin_stability_score"] if basin else 0.5

        # Lyapunov max
        lyap_max = lyapunov["lyapunov_max"] if lyapunov else None

        # Current regime
        current_regime = regime["current_regime"] if regime else "UNKNOWN"

        # Birth grade
        birth_grade = birth["birth_grade"] if birth else None

        # Prognosis
        if birth:
            prognosis = birth["prognosis"]
        elif stability_class == "CHAOTIC":
            prognosis = "System showing chaotic dynamics. Immediate attention required."
        elif stability_class == "UNSTABLE":
            prognosis = "System unstable. Monitor closely for regime transitions."
        elif stability_class == "MARGINAL":
            prognosis = "System marginally stable. Watch for destabilization."
        else:
            prognosis = "System stable. Normal operation."

        return StabilityDiagnosis(
            stability_class=stability_class,
            basin_score=basin_score,
            lyapunov_max=lyap_max,
            regime=current_regime,
            birth_grade=birth_grade,
            prognosis=prognosis,
        )

    def get_fleet_stability_summary(self) -> Dict[str, Any]:
        """Get stability summary for entire fleet."""
        entities = self.get_entities()

        results = []
        stability_counts = {"STABLE": 0, "MARGINAL": 0, "UNSTABLE": 0, "CHAOTIC": 0}
        birth_grades = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0}

        for entity in entities:
            diagnosis = self.analyze_stability(entity)

            stability_counts[diagnosis.stability_class] = stability_counts.get(diagnosis.stability_class, 0) + 1
            if diagnosis.birth_grade:
                birth_grades[diagnosis.birth_grade] = birth_grades.get(diagnosis.birth_grade, 0) + 1

            results.append({
                "entity_id": entity,
                "stability_class": diagnosis.stability_class,
                "basin_score": diagnosis.basin_score,
                "regime": diagnosis.regime,
                "birth_grade": diagnosis.birth_grade,
            })

        return {
            "n_entities": len(entities),
            "stability_counts": stability_counts,
            "birth_grade_counts": birth_grades,
            "pct_unstable": (stability_counts["UNSTABLE"] + stability_counts["CHAOTIC"]) / len(entities) * 100 if entities else 0,
            "entities": sorted(results, key=lambda x: x["basin_score"]),
        }


# =============================================================================
# STORY TEMPLATES
# =============================================================================

STABILITY_STORIES = {
    "DESTABILIZING": """
{entity_id} shows increasing dynamical instability.

Current regime: {regime}
Coherence trend: {coherence_trend:+.4f}/cycle
Velocity trend: {velocity_trend:+.4f}/cycle
Basin stability: {basin_score:.0%}

INTERPRETATION: The system's operating attractor is becoming shallower.
Small perturbations will have larger effects.

WATCH FOR: Increased variability, mode jumping, intermittent behavior.
""",

    "CHAOTIC": """
{entity_id} is exhibiting CHAOTIC dynamics.

Lyapunov exponent: {lyapunov_max:.3f} (positive = chaos)
{n_chaotic_signals}/{n_signals} signals show chaotic behavior
Basin stability: {basin_score:.0%}

INTERPRETATION: Trajectories are diverging exponentially.
The system has lost predictable behavior.

ACTION: Immediate investigation required. This may indicate:
- Advanced degradation
- Operating outside design envelope
- Control system instability
""",

    "BIRTH_CERTIFICATE": """
Early-life stability analysis for {entity_id}:

Birth Certificate Grade: {birth_grade}
Birth Certificate Score: {birth_certificate_score:.0%}

Early-life metrics (first {early_pct:.0%}):
  Coherence: {early_coherence:.3f}
  Velocity:  {early_velocity:.4f}
  Volatility: {early_volatility:.4f}

PROGNOSIS: {prognosis}

Based on fleet data, systems with similar early profiles have:
  Average lifespan: {expected_lifespan} cycles
  Survival rate to {survival_threshold}: {survival_rate:.0%}
""",

    "STABLE": """
{entity_id} is operating in a STABLE regime.

Basin stability: {basin_score:.0%}
Current coherence: {recent_coherence:.3f}
Current velocity: {recent_velocity:.4f}

INTERPRETATION: System is well-attracted to its operating point.
Normal operation expected.

Birth grade: {birth_grade}
""",
}


def generate_stability_story(interpreter: DynamicsInterpreter, entity_id: str) -> str:
    """Generate narrative for entity stability."""
    diagnosis = interpreter.analyze_stability(entity_id)
    basin = interpreter.compute_basin_stability(entity_id)
    regime = interpreter.detect_regime(entity_id)
    birth = interpreter.compute_birth_certificate(entity_id)
    lyapunov = interpreter.analyze_lyapunov(entity_id)

    # Choose template based on state
    if diagnosis.stability_class == "CHAOTIC" and lyapunov:
        template = STABILITY_STORIES["CHAOTIC"]
        return template.format(
            entity_id=entity_id,
            lyapunov_max=lyapunov["lyapunov_max"],
            n_chaotic_signals=lyapunov["n_chaotic_signals"],
            n_signals=lyapunov["n_signals"],
            basin_score=diagnosis.basin_score,
        )

    elif regime and regime["current_regime"] == "DESTABILIZING":
        template = STABILITY_STORIES["DESTABILIZING"]
        return template.format(
            entity_id=entity_id,
            regime=regime["current_regime"],
            coherence_trend=regime["coherence_trend"],
            velocity_trend=regime["velocity_trend"],
            basin_score=diagnosis.basin_score,
        )

    elif birth:
        template = STABILITY_STORIES["BIRTH_CERTIFICATE"]
        return template.format(
            entity_id=entity_id,
            birth_grade=birth["birth_grade"],
            birth_certificate_score=birth["birth_certificate_score"],
            early_pct=0.2,
            early_coherence=birth["early_coherence"],
            early_velocity=birth["early_velocity"],
            early_volatility=birth["early_volatility"],
            prognosis=birth["prognosis"],
            expected_lifespan="N/A",  # Would need fleet stats
            survival_threshold="N/A",
            survival_rate=0,
        )

    else:
        template = STABILITY_STORIES["STABLE"]
        return template.format(
            entity_id=entity_id,
            basin_score=diagnosis.basin_score,
            recent_coherence=regime["recent_coherence"] if regime else 0,
            recent_velocity=regime["recent_velocity"] if regime else 0,
            birth_grade=diagnosis.birth_grade or "N/A",
        )


__all__ = [
    "DynamicsInterpreter",
    "StabilityDiagnosis",
    "generate_stability_story",
    "STABILITY_STORIES",
]
