"""
Energy Dynamics Interpreter
===========================

Interprets eigenstructure metrics from PRISM as energy dynamics.

Physical analogy:
- total_variance = system energy (how much "activity" in the state space)
- concentration_ratio = energy storage (how much in dominant mode)
- eigenvalue_entropy = energy distribution (even vs concentrated)
- pc1_alignment = mode stability (how stable the dominant mode is)

Energy states:
- ACCUMULATING: Energy building up (pre-failure signature)
- DISSIPATING: Energy leaving (return to baseline)
- STABLE: Energy balanced
- REDISTRIBUTING: Modes shifting without energy change
"""

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum


class EnergyState(Enum):
    """Energy state classifications based on effective dimension dynamics."""
    COLLAPSING = "COLLAPSING"          # Dimension decreasing (pre-failure)
    EXPANDING = "EXPANDING"            # Dimension increasing (recovery)
    STABLE = "STABLE"                  # Dimension balanced
    REDISTRIBUTING = "REDISTRIBUTING"  # Mode shift without dimension change
    UNKNOWN = "UNKNOWN"


@dataclass
class EnergyDiagnosis:
    """Diagnosis of energy dynamics state."""
    energy_state: str            # ACCUMULATING, DISSIPATING, STABLE, REDISTRIBUTING
    energy_level: str            # HIGH, NORMAL, LOW
    concentration: str           # FOCUSED, DISTRIBUTED
    mode_stability: str          # STABLE, SHIFTING, CHAOTIC
    energy_trend: float          # Positive = accumulating
    risk_score: float            # 0-1, higher = more concern
    interpretation: str          # Human-readable interpretation


class EnergyInterpreter:
    """
    Interprets eigenstructure metrics as energy dynamics.

    Uses state_geometry.parquet from PRISM to understand:
    - System energy level (total_variance)
    - Energy concentration (eigenvalue ratios)
    - Mode stability (alignment changes)
    - Energy flow (variance derivatives)
    """

    # Thresholds
    HIGH_ENERGY_THRESHOLD = 0.7      # Normalized energy above this = HIGH
    LOW_ENERGY_THRESHOLD = 0.3       # Normalized energy below this = LOW
    CONCENTRATION_THRESHOLD = 0.6    # Concentration ratio above this = FOCUSED
    STABLE_ALIGNMENT_THRESHOLD = 0.8 # Alignment above this = STABLE modes
    ENERGY_TREND_THRESHOLD = 0.05    # |trend| above this = significant change

    def __init__(
        self,
        prism_output: Optional[str] = None,
        state_geometry_path: Optional[str] = None,
        state_geometry_df: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize interpreter.

        Args:
            prism_output: Path to PRISM output directory
            state_geometry_path: Path to state_geometry.parquet
            state_geometry_df: Or provide DataFrame directly
        """
        if prism_output:
            output_dir = Path(prism_output)
            if state_geometry_path is None:
                state_geometry_path = output_dir / "state_geometry.parquet"

        if state_geometry_path and not isinstance(state_geometry_path, Path):
            state_geometry_path = Path(state_geometry_path)

        if state_geometry_df is not None:
            self.state_geometry = state_geometry_df
        elif state_geometry_path and state_geometry_path.exists():
            self.state_geometry = pl.read_parquet(state_geometry_path)
        else:
            self.state_geometry = None

        # Cache for normalization stats
        self._energy_stats: Optional[Dict] = None

    def _compute_energy_stats(self) -> Dict[str, float]:
        """Compute fleet-wide energy statistics for normalization.

        Uses effective_dim as primary energy metric (not total_variance which
        is often constant in scale-invariant PRISM).
        """
        if self._energy_stats is not None:
            return self._energy_stats

        if self.state_geometry is None:
            return {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 4.0}

        # Use shape engine for energy metrics
        shape_data = self.state_geometry.filter(pl.col("engine") == "shape")

        if shape_data.is_empty():
            return {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 4.0}

        # Use effective_dim as primary energy metric
        eff_dim = shape_data["effective_dim"].drop_nulls()

        self._energy_stats = {
            "mean": float(eff_dim.mean()) if len(eff_dim) > 0 else 2.0,
            "std": float(eff_dim.std()) if len(eff_dim) > 1 else 0.5,
            "min": float(eff_dim.min()) if len(eff_dim) > 0 else 1.0,
            "max": float(eff_dim.max()) if len(eff_dim) > 0 else 4.0,
        }

        return self._energy_stats

    def get_cohorts(self) -> List[str]:
        """Get list of cohort IDs."""
        if self.state_geometry is None:
            return []

        if "cohort" in self.state_geometry.columns:
            return self.state_geometry["cohort"].unique().sort().to_list()
        return ["default"]

    def compute_energy_metrics(self, cohort: str) -> Optional[Dict[str, Any]]:
        """
        Compute energy metrics for a cohort.

        Note: In scale-invariant PRISM, total_variance is often constant (normalized).
        We use effective_dim as the primary energy metric:
        - Higher effective_dim = energy distributed across modes (healthy)
        - Lower effective_dim = energy concentrated (dimension collapse = stress)

        Returns energy interpretations:
        - effective_dim: Primary energy metric (higher = distributed, lower = concentrated)
        - concentration_ratio: Energy in dominant mode
        - mode_stability: Stability of dominant mode (pc1_alignment)
        """
        if self.state_geometry is None:
            return None

        # Filter to cohort and shape engine
        has_cohort = "cohort" in self.state_geometry.columns
        if has_cohort:
            cohort_data = self.state_geometry.filter(
                (pl.col("cohort") == cohort) & (pl.col("engine") == "shape")
            ).sort("I")
        else:
            cohort_data = self.state_geometry.filter(
                pl.col("engine") == "shape"
            ).sort("I")

        if cohort_data.is_empty():
            return None

        # Extract metrics - use effective_dim as primary energy metric
        eff_dim = cohort_data["effective_dim"].to_numpy()
        total_variance = cohort_data["total_variance"].to_numpy()

        # Compute concentration ratio: eigenvalue_1 / total_variance
        if "eigenvalue_1" in cohort_data.columns:
            eig1 = cohort_data["eigenvalue_1"].to_numpy()
            concentration = np.where(
                total_variance > 0,
                eig1 / total_variance,
                0.5
            )
        else:
            # Estimate from effective_dim (inverse relationship)
            concentration = 1.0 / np.maximum(eff_dim, 1.0)

        # Mode stability from alignment (how stable is the dominant mode)
        if "pc1_alignment" in cohort_data.columns:
            alignment = cohort_data["pc1_alignment"].to_numpy()
        else:
            alignment = np.ones(len(concentration)) * 0.5

        # Normalize effective_dim to 0-1 scale based on fleet stats
        stats = self._compute_energy_stats()
        if stats["max"] > stats["min"]:
            normalized_energy = (eff_dim - stats["min"]) / (stats["max"] - stats["min"])
        else:
            normalized_energy = np.ones(len(eff_dim)) * 0.5

        # Compute effective_dim trend (collapse = decreasing eff_dim = negative trend)
        window = max(2, min(5, len(eff_dim) // 3))
        if len(eff_dim) >= window * 2:
            recent_dim = np.mean(eff_dim[-window:])
            earlier_dim = np.mean(eff_dim[:window])  # Compare to early life
            # Negative = collapsing (concentrating), Positive = expanding (distributing)
            dim_change = (recent_dim - earlier_dim) / max(earlier_dim, 1e-10)
        else:
            dim_change = 0.0

        # Alignment trend (stability change)
        valid_alignment = alignment[~np.isnan(alignment)]
        if len(valid_alignment) >= window * 2:
            recent_align = np.mean(valid_alignment[-window:])
            earlier_align = np.mean(valid_alignment[:window])
            alignment_change = recent_align - earlier_align
        else:
            alignment_change = 0.0

        return {
            "mean_effective_dim": float(np.mean(eff_dim)),
            "normalized_energy": float(np.mean(normalized_energy)),
            "effective_dim_std": float(np.std(eff_dim)),
            "mean_concentration": float(np.mean(concentration)),
            "mean_alignment": float(np.nanmean(alignment)),
            "alignment_std": float(np.nanstd(alignment)),
            "dim_trend": dim_change,           # Negative = collapse
            "alignment_trend": alignment_change,  # Stability change
            "n_observations": len(eff_dim),
            "final_effective_dim": float(eff_dim[-1]) if len(eff_dim) > 0 else 0.0,
            "final_concentration": float(concentration[-1]) if len(concentration) > 0 else 0.5,
            "final_alignment": float(alignment[-1]) if len(alignment) > 0 and not np.isnan(alignment[-1]) else 0.5,
        }

    def classify_energy_state(self, cohort: str) -> EnergyDiagnosis:
        """
        Classify the energy state of a cohort.

        Energy interpretation for scale-invariant PRISM:
        - effective_dim is the primary energy metric
        - Decreasing effective_dim = dimension COLLAPSE (energy concentrating = stress)
        - Increasing effective_dim = dimension EXPANSION (energy distributing = recovery)

        Energy states:
        - COLLAPSING: Effective dim decreasing (pre-failure signature)
        - EXPANDING: Effective dim increasing (recovery/stabilization)
        - STABLE: Effective dim balanced
        - REDISTRIBUTING: Modes shifting without clear collapse/expansion
        """
        metrics = self.compute_energy_metrics(cohort)

        if metrics is None:
            return EnergyDiagnosis(
                energy_state="UNKNOWN",
                energy_level="UNKNOWN",
                concentration="UNKNOWN",
                mode_stability="UNKNOWN",
                energy_trend=0.0,
                risk_score=0.5,
                interpretation="Insufficient data for energy analysis"
            )

        # Classify energy level (based on effective_dim relative to fleet)
        norm_energy = metrics["normalized_energy"]
        if norm_energy > self.HIGH_ENERGY_THRESHOLD:
            energy_level = "DISTRIBUTED"  # High effective_dim
        elif norm_energy < self.LOW_ENERGY_THRESHOLD:
            energy_level = "CONCENTRATED"  # Low effective_dim = collapsed
        else:
            energy_level = "NORMAL"

        # Classify concentration (from eigenvalue ratio)
        if metrics["mean_concentration"] > self.CONCENTRATION_THRESHOLD:
            concentration = "FOCUSED"  # Energy in few modes
        else:
            concentration = "DISTRIBUTED"

        # Classify mode stability (from alignment std)
        align_std = metrics.get("alignment_std", 0.2)
        mean_align = metrics["mean_alignment"]
        if align_std < 0.15 and mean_align > 0.6:
            mode_stability = "STABLE"
        elif align_std > 0.3 or mean_align < 0.3:
            mode_stability = "CHAOTIC"
        else:
            mode_stability = "SHIFTING"

        # Classify energy state from dim_trend (key metric)
        # Negative dim_trend = COLLAPSING (losing dimensions = stress)
        # Positive dim_trend = EXPANDING (gaining dimensions = recovery)
        dim_trend = metrics["dim_trend"]
        alignment_trend = metrics.get("alignment_trend", 0.0)

        if abs(dim_trend) < self.ENERGY_TREND_THRESHOLD:
            if abs(alignment_trend) > 0.1:
                energy_state = "REDISTRIBUTING"
            else:
                energy_state = "STABLE"
        elif dim_trend < 0:
            energy_state = "COLLAPSING"  # Dimension collapse = pre-failure
        else:
            energy_state = "EXPANDING"   # Dimension expansion = recovery

        # Compute risk score
        # Collapsing + concentrated + chaotic modes = higher risk
        risk_score = 0.0

        # Dimension level contribution
        if energy_level == "CONCENTRATED":
            risk_score += 0.35  # Already collapsed = high risk
        elif energy_level == "DISTRIBUTED":
            risk_score += 0.10  # Healthy
        else:
            risk_score += 0.20

        # Trend contribution
        if energy_state == "COLLAPSING":
            risk_score += 0.40  # Active collapse = highest risk
        elif energy_state == "REDISTRIBUTING":
            risk_score += 0.20
        elif energy_state == "EXPANDING":
            risk_score += 0.05  # Recovery = low risk
        else:
            risk_score += 0.15

        # Mode stability contribution
        if mode_stability == "CHAOTIC":
            risk_score += 0.25
        elif mode_stability == "SHIFTING":
            risk_score += 0.15
        else:
            risk_score += 0.05

        # Generate interpretation
        interpretation = self._generate_interpretation(
            energy_state, energy_level, concentration, mode_stability, dim_trend
        )

        return EnergyDiagnosis(
            energy_state=energy_state,
            energy_level=energy_level,
            concentration=concentration,
            mode_stability=mode_stability,
            energy_trend=dim_trend,  # Use dim_trend as primary trend
            risk_score=min(risk_score, 1.0),
            interpretation=interpretation
        )

    def _generate_interpretation(
        self,
        energy_state: str,
        energy_level: str,
        concentration: str,
        mode_stability: str,
        dim_trend: float
    ) -> str:
        """Generate human-readable interpretation."""

        parts = []

        # Dimension level interpretation
        if energy_level == "CONCENTRATED":
            parts.append("Effective dimension is below fleet baseline - system state is collapsed.")
        elif energy_level == "DISTRIBUTED":
            parts.append("Effective dimension is above fleet baseline - healthy mode distribution.")
        else:
            parts.append("Effective dimension is within normal range.")

        # Energy state interpretation (based on dim_trend)
        if energy_state == "COLLAPSING":
            parts.append(f"Dimension is COLLAPSING ({dim_trend:+.1%} trend).")
            parts.append("This is a pre-failure signature - degrees of freedom are being lost.")
        elif energy_state == "EXPANDING":
            parts.append(f"Dimension is expanding ({dim_trend:+.1%} trend).")
            parts.append("System appears to be recovering or gaining stability.")
        elif energy_state == "REDISTRIBUTING":
            parts.append("Modes are redistributing without net dimension change.")
            parts.append("The system may be adapting to new operating conditions.")
        else:
            parts.append("Dimension dynamics are stable.")

        # Concentration interpretation
        if concentration == "FOCUSED":
            parts.append("Energy is concentrated in dominant modes.")
        else:
            parts.append("Energy is distributed across multiple modes.")

        # Mode stability interpretation
        if mode_stability == "CHAOTIC":
            parts.append("Mode structure is unstable - unpredictable behavior likely.")
        elif mode_stability == "SHIFTING":
            parts.append("Mode alignment is shifting - monitor for regime transitions.")

        return " ".join(parts)

    def analyze_energy_trajectory(self, cohort: str) -> Optional[Dict[str, Any]]:
        """
        Analyze energy trajectory over the full lifecycle.

        Uses effective_dim as primary energy metric (dimension collapse = pre-failure).

        Returns:
        - effective_dim trajectory over time
        - collapse_onset: When dimension started collapsing (negative trend)
        - phase_transitions: Detected collapse/expansion transitions
        """
        if self.state_geometry is None:
            return None

        has_cohort = "cohort" in self.state_geometry.columns
        if has_cohort:
            cohort_data = self.state_geometry.filter(
                (pl.col("cohort") == cohort) & (pl.col("engine") == "shape")
            ).sort("I")
        else:
            cohort_data = self.state_geometry.filter(
                pl.col("engine") == "shape"
            ).sort("I")

        if cohort_data.is_empty():
            return None

        I_values = cohort_data["I"].to_numpy()
        eff_dim = cohort_data["effective_dim"].to_numpy()

        # Also get alignment for mode stability tracking
        if "pc1_alignment" in cohort_data.columns:
            alignment = cohort_data["pc1_alignment"].to_numpy()
        else:
            alignment = np.ones(len(eff_dim)) * 0.5

        # Compute rolling dimension trend (negative = collapse)
        window = max(2, len(eff_dim) // 5)
        dim_trend = np.zeros(len(eff_dim))
        for i in range(window, len(eff_dim)):
            dim_trend[i] = (eff_dim[i] - eff_dim[i-window]) / max(eff_dim[i-window], 1e-10)

        # Detect collapse onset (first sustained negative trend)
        collapse_onset = None
        for i in range(window, len(dim_trend) - window):
            if np.all(dim_trend[i:i+window] < -self.ENERGY_TREND_THRESHOLD):
                collapse_onset = int(I_values[i])
                break

        # Detect phase transitions (sign changes in trend)
        transitions = []
        for i in range(1, len(dim_trend)):
            if dim_trend[i-1] * dim_trend[i] < 0 and abs(dim_trend[i]) > self.ENERGY_TREND_THRESHOLD:
                transition_type = "expanding" if dim_trend[i] > 0 else "collapsing"
                transitions.append({
                    "I": int(I_values[i]),
                    "type": transition_type,
                    "effective_dim": float(eff_dim[i])
                })

        return {
            "I": I_values.tolist(),
            "effective_dim": eff_dim.tolist(),
            "alignment": [float(a) if not np.isnan(a) else None for a in alignment],
            "dim_trend": dim_trend.tolist(),
            "collapse_onset": collapse_onset,
            "n_transitions": len(transitions),
            "transitions": transitions[:10],  # Limit to first 10
            "lifecycle_length": len(I_values),
            "initial_dim": float(eff_dim[0]) if len(eff_dim) > 0 else None,
            "final_dim": float(eff_dim[-1]) if len(eff_dim) > 0 else None,
            "dim_change": float(eff_dim[-1] - eff_dim[0]) if len(eff_dim) > 1 else 0.0,
        }

    def get_fleet_energy_summary(self) -> Dict[str, Any]:
        """Get energy summary for entire fleet."""
        cohorts = self.get_cohorts()

        results = []
        state_counts = {s.value: 0 for s in EnergyState}

        for cohort in cohorts:
            diagnosis = self.classify_energy_state(cohort)
            state_counts[diagnosis.energy_state] = state_counts.get(diagnosis.energy_state, 0) + 1

            metrics = self.compute_energy_metrics(cohort)

            results.append({
                "cohort": cohort,
                "energy_state": diagnosis.energy_state,
                "energy_level": diagnosis.energy_level,
                "risk_score": diagnosis.risk_score,
                "energy_trend": diagnosis.energy_trend,
                "mean_effective_dim": metrics["mean_effective_dim"] if metrics else None,
            })

        # Sort by risk score descending
        results.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "n_cohorts": len(cohorts),
            "state_counts": state_counts,
            "pct_collapsing": state_counts.get("COLLAPSING", 0) / len(cohorts) * 100 if cohorts else 0,
            "pct_high_risk": len([r for r in results if r["risk_score"] > 0.6]) / len(cohorts) * 100 if cohorts else 0,
            "cohorts": results,
        }


# =============================================================================
# STORY TEMPLATES
# =============================================================================

ENERGY_STORIES = {
    "COLLAPSING": """
{cohort} is showing DIMENSION COLLAPSE.

Current state:
  Effective dim: {mean_effective_dim:.2f} (normalized: {normalized_energy:.0%})
  Dimension trend: {energy_trend:+.1%} (DECREASING)
  Concentration: {concentration} ({mean_concentration:.1%} in dominant mode)
  Mode stability: {mode_stability}

INTERPRETATION: The system is LOSING degrees of freedom. This is a
critical pre-failure signature indicating:
- State space contracting
- Independent modes becoming coupled
- System approaching a failure attractor

WATCH FOR: Accelerating collapse, mode lock-in, or sudden transition.
Risk score: {risk_score:.0%}
""",

    "EXPANDING": """
{cohort} is showing DIMENSION EXPANSION.

Current state:
  Effective dim: {mean_effective_dim:.2f} (normalized: {normalized_energy:.0%})
  Dimension trend: {energy_trend:+.1%} (INCREASING)
  Concentration: {concentration}
  Mode stability: {mode_stability}

INTERPRETATION: The system is GAINING degrees of freedom. This typically indicates:
- Recovery from a constrained state
- Return to healthy baseline operation
- Successful intervention or reduced stress

PROGNOSIS: System is recovering. Continue monitoring for stabilization.
Risk score: {risk_score:.0%}
""",

    "REDISTRIBUTING": """
{cohort} is showing MODE REDISTRIBUTION.

Current state:
  Effective dim: {mean_effective_dim:.2f} (normalized: {normalized_energy:.0%})
  Dimension trend: {energy_trend:+.1%} (stable)
  Concentration: {concentration} (shifting)
  Mode stability: {mode_stability}

INTERPRETATION: Total effective dimension is stable but modes are reshuffling.
This may indicate:
- Adaptation to new operating conditions
- Load redistribution among subsystems
- Early warning of impending collapse

WATCH FOR: Whether new mode structure stabilizes or begins collapsing.
Risk score: {risk_score:.0%}
""",

    "STABLE": """
{cohort} is in a STABLE dimensional state.

Current state:
  Effective dim: {mean_effective_dim:.2f} (normalized: {normalized_energy:.0%})
  Dimension trend: {energy_trend:+.1%}
  Concentration: {concentration}
  Mode stability: {mode_stability}

INTERPRETATION: Dimensional dynamics are balanced. System is operating
within expected parameters with healthy mode distribution.

Risk score: {risk_score:.0%}
""",
}


def generate_energy_story(interpreter: EnergyInterpreter, cohort: str) -> str:
    """Generate narrative for cohort energy state."""
    diagnosis = interpreter.classify_energy_state(cohort)
    metrics = interpreter.compute_energy_metrics(cohort)

    if metrics is None:
        return f"Insufficient data to analyze energy dynamics for {cohort}."

    template = ENERGY_STORIES.get(diagnosis.energy_state, ENERGY_STORIES["STABLE"])

    return template.format(
        cohort=cohort,
        energy_level=diagnosis.energy_level,
        mean_effective_dim=metrics["mean_effective_dim"],
        normalized_energy=metrics["normalized_energy"],
        energy_trend=diagnosis.energy_trend,
        concentration=diagnosis.concentration,
        mean_concentration=metrics["mean_concentration"],
        mode_stability=diagnosis.mode_stability,
        risk_score=diagnosis.risk_score,
    )


__all__ = [
    "EnergyInterpreter",
    "EnergyDiagnosis",
    "EnergyState",
    "generate_energy_story",
    "ENERGY_STORIES",
]
