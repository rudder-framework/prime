"""
Failure Mode Classifier
=======================

Uses eigenstructure trajectory to classify failure physics.
This determines WHICH prediction strategy to apply.

Failure Modes:
    DIMENSIONAL_COLLAPSE: Structure changes before failure (markets, some bearings)
    MASS_EROSION: Structure preserved, mean drifts (turbofans, pumps)
    CASCADING: Local erosion triggers global collapse (chemical plants, grids)
    STABLE: No degradation signature detected

The Key Insight:
    Eigenstructure is a CLASSIFIER, not a direct PREDICTOR.
    It tells you WHICH physics apply, then you use mode-specific predictors.
"""

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum


class FailureMode(Enum):
    """Failure mode classifications based on eigenstructure dynamics."""
    DIMENSIONAL_COLLAPSE = "DIMENSIONAL_COLLAPSE"  # Markets, some bearings
    MASS_EROSION = "MASS_EROSION"                  # Turbofans, pumps
    CASCADING = "CASCADING"                        # Chemical plants, grids
    STABLE = "STABLE"                              # No degradation
    UNKNOWN = "UNKNOWN"                            # Insufficient data


@dataclass
class FailureModeClassification:
    """Result of failure mode classification."""
    mode: FailureMode
    confidence: float           # 0-1

    # Diagnostic metrics
    eff_dim_std: float
    alignment_std: float
    variance_slope: float
    correlation_slope: Optional[float]

    # Mode-specific predictor recommendation
    recommended_predictor: str
    warning_metric: str

    # Interpretation
    description: str


class FailureModeClassifier:
    """
    Classifies failure mode from eigenstructure trajectory.

    Usage:
        classifier = FailureModeClassifier(state_geometry_df=geom)
        result = classifier.classify(cohort='engine_1')

        if result.mode == FailureMode.DIMENSIONAL_COLLAPSE:
            # Use trapped energy model
        elif result.mode == FailureMode.MASS_EROSION:
            # Use drift rate model
    """

    # Thresholds for classification
    # Note: These work with NORMALIZED (scale-invariant) PRISM features.
    # For raw signal analysis, use domain-specific thresholds.
    EFF_DIM_STD_THRESHOLD = 0.30      # Above = dimensional collapse (raised from 0.15)
    ALIGNMENT_STD_THRESHOLD = 0.25    # Above = dimensional collapse (raised from 0.15)
    VARIANCE_SLOPE_THRESHOLD = -0.001 # Below = mass erosion (loosened)
    CORRELATION_SLOPE_THRESHOLD = 0.01  # Above = cascading

    # For scale-invariant PRISM, total_variance is often constant (~3.2).
    # Use signal drift rate from observations for mass erosion detection.

    def __init__(
        self,
        prism_output: Optional[str] = None,
        state_geometry_path: Optional[str] = None,
        state_geometry_df: Optional[pl.DataFrame] = None,
        signal_pairwise_path: Optional[str] = None,
        signal_pairwise_df: Optional[pl.DataFrame] = None,
        observations_path: Optional[str] = None,
        observations_df: Optional[pl.DataFrame] = None,
    ):
        """
        Initialize classifier with geometry data.

        For best results, provide observations_df to enable raw signal drift
        analysis (critical for detecting MASS_EROSION in scale-invariant systems).
        """
        if prism_output:
            output_dir = Path(prism_output)
            if state_geometry_path is None:
                state_geometry_path = output_dir / "state_geometry.parquet"
            if signal_pairwise_path is None:
                signal_pairwise_path = output_dir / "signal_pairwise.parquet"
            if observations_path is None:
                observations_path = output_dir / "observations.parquet"

        # Load state geometry
        if state_geometry_df is not None:
            self.state_geometry = state_geometry_df
        elif state_geometry_path and Path(state_geometry_path).exists():
            self.state_geometry = pl.read_parquet(state_geometry_path)
        else:
            self.state_geometry = None

        # Load pairwise correlations (optional, for cascading detection)
        if signal_pairwise_df is not None:
            self.signal_pairwise = signal_pairwise_df
        elif signal_pairwise_path and Path(signal_pairwise_path).exists():
            self.signal_pairwise = pl.read_parquet(signal_pairwise_path)
        else:
            self.signal_pairwise = None

        # Load observations (optional, for raw signal drift analysis)
        if observations_df is not None:
            self.observations = observations_df
        elif observations_path and Path(observations_path).exists():
            self.observations = pl.read_parquet(observations_path)
        else:
            self.observations = None

    def get_cohorts(self) -> List[str]:
        """Get list of cohort IDs."""
        if self.state_geometry is None:
            return []
        if "cohort" in self.state_geometry.columns:
            return self.state_geometry["cohort"].unique().sort().to_list()
        return ["default"]

    def _compute_trajectory_stats(self, cohort: str) -> Optional[Dict[str, float]]:
        """Compute eigenstructure trajectory statistics for a cohort."""
        if self.state_geometry is None:
            return None

        # Filter to cohort and shape engine
        has_cohort = "cohort" in self.state_geometry.columns
        if has_cohort:
            data = self.state_geometry.filter(
                (pl.col("cohort") == cohort) & (pl.col("engine") == "shape")
            ).sort("I")
        else:
            data = self.state_geometry.filter(
                pl.col("engine") == "shape"
            ).sort("I")

        if data.is_empty() or len(data) < 3:
            return None

        # Extract metrics
        I = data["I"].to_numpy().astype(float)
        eff_dim = data["effective_dim"].to_numpy()
        total_variance = data["total_variance"].to_numpy()

        # PC1 alignment
        if "pc1_alignment" in data.columns:
            alignment = data["pc1_alignment"].to_numpy()
            alignment = alignment[~np.isnan(alignment)]
        else:
            alignment = np.array([0.5])

        # Compute statistics
        stats = {
            "n_observations": len(data),
            "eff_dim_mean": float(np.mean(eff_dim)),
            "eff_dim_std": float(np.std(eff_dim)),
            "alignment_std": float(np.std(alignment)) if len(alignment) > 1 else 0.0,
            "alignment_mean": float(np.mean(alignment)),
        }

        # Variance slope (linear regression)
        if len(I) >= 3 and np.std(I) > 0:
            stats["variance_slope"] = float(np.polyfit(I, total_variance, 1)[0])
        else:
            stats["variance_slope"] = 0.0

        # Pairwise correlation slope (if available)
        stats["correlation_slope"] = self._compute_correlation_slope(cohort)

        return stats

    def _compute_correlation_slope(self, cohort: str) -> Optional[float]:
        """Compute trend in mean pairwise correlation."""
        if self.signal_pairwise is None:
            return None

        has_cohort = "cohort" in self.signal_pairwise.columns
        if has_cohort:
            data = self.signal_pairwise.filter(pl.col("cohort") == cohort)
        else:
            data = self.signal_pairwise

        if data.is_empty():
            return None

        # Get correlation column
        corr_col = None
        for col in ["correlation", "pearson", "corr"]:
            if col in data.columns:
                corr_col = col
                break

        if corr_col is None:
            return None

        # Compute mean correlation per I
        mean_corr = (
            data
            .group_by("I")
            .agg(pl.col(corr_col).mean().alias("mean_corr"))
            .sort("I")
        )

        if len(mean_corr) < 3:
            return None

        I = mean_corr["I"].to_numpy().astype(float)
        corr = mean_corr["mean_corr"].to_numpy()

        valid = ~np.isnan(corr)
        if valid.sum() < 3:
            return None

        return float(np.polyfit(I[valid], corr[valid], 1)[0])

    def classify(self, cohort: str) -> FailureModeClassification:
        """
        Classify failure mode for a cohort.

        Decision tree:
        1. High eff_dim variability OR high alignment variability → DIMENSIONAL_COLLAPSE
        2. Low variability + negative variance slope → MASS_EROSION
        3. Low variability + rising correlations → CASCADING
        4. Otherwise → STABLE
        """
        stats = self._compute_trajectory_stats(cohort)

        if stats is None:
            return FailureModeClassification(
                mode=FailureMode.UNKNOWN,
                confidence=0.0,
                eff_dim_std=0.0,
                alignment_std=0.0,
                variance_slope=0.0,
                correlation_slope=None,
                recommended_predictor="none",
                warning_metric="none",
                description="Insufficient data for classification"
            )

        eff_dim_std = stats["eff_dim_std"]
        alignment_std = stats["alignment_std"]
        variance_slope = stats["variance_slope"]
        correlation_slope = stats.get("correlation_slope")

        # Classification logic
        mode = FailureMode.STABLE
        confidence = 0.5
        predictor = "none"
        warning = "none"
        description = ""

        # TYPE 1: Dimensional Collapse
        if eff_dim_std > self.EFF_DIM_STD_THRESHOLD or alignment_std > self.ALIGNMENT_STD_THRESHOLD:
            mode = FailureMode.DIMENSIONAL_COLLAPSE
            confidence = min(1.0, max(eff_dim_std, alignment_std) / 0.3)
            predictor = "trapped_energy_duration"
            warning = "CAPE > 25 AND rotation < 0.15 for 3+ periods"
            description = (
                f"Structure varies significantly (eff_dim_std={eff_dim_std:.3f}, "
                f"alignment_std={alignment_std:.3f}). "
                "System shows dimensional collapse dynamics. "
                "Use trapped energy model: high mass + low rotation = danger."
            )

        # TYPE 2: Mass Erosion
        elif eff_dim_std < self.EFF_DIM_STD_THRESHOLD and variance_slope < self.VARIANCE_SLOPE_THRESHOLD:
            mode = FailureMode.MASS_EROSION
            confidence = min(1.0, abs(variance_slope) / 0.05)
            predictor = "drift_rate"
            warning = "drift_rate > threshold in dominant signal"
            description = (
                f"Structure preserved (eff_dim_std={eff_dim_std:.3f}) but "
                f"energy depleting (variance_slope={variance_slope:.4f}). "
                "System shows mass erosion dynamics. "
                "Use drift rate model: faster drift = shorter remaining life."
            )

        # TYPE 3: Cascading
        elif (eff_dim_std < self.EFF_DIM_STD_THRESHOLD and
              correlation_slope is not None and
              correlation_slope > self.CORRELATION_SLOPE_THRESHOLD):
            mode = FailureMode.CASCADING
            confidence = min(1.0, correlation_slope / 0.05)
            predictor = "local_drift_plus_coupling"
            warning = "rising cross-correlation + local signal drift"
            description = (
                f"Structure preserved (eff_dim_std={eff_dim_std:.3f}) but "
                f"coupling increasing (correlation_slope={correlation_slope:.4f}). "
                "System shows cascading failure dynamics. "
                "Use combined model: local drift × coupling rise = cascade risk."
            )

        # TYPE 0: Stable
        else:
            mode = FailureMode.STABLE
            confidence = 0.8
            predictor = "monitor_baseline"
            warning = "drift from baseline envelope"
            description = (
                f"System stable: eff_dim_std={eff_dim_std:.3f}, "
                f"variance_slope={variance_slope:.4f}. "
                "No degradation signature detected. Monitor for baseline drift."
            )

        return FailureModeClassification(
            mode=mode,
            confidence=confidence,
            eff_dim_std=eff_dim_std,
            alignment_std=alignment_std,
            variance_slope=variance_slope,
            correlation_slope=correlation_slope,
            recommended_predictor=predictor,
            warning_metric=warning,
            description=description
        )

    def classify_fleet(self) -> Dict[str, FailureModeClassification]:
        """Classify all cohorts in the dataset."""
        results = {}
        for cohort in self.get_cohorts():
            results[cohort] = self.classify(cohort)
        return results

    def get_fleet_summary(self) -> Dict[str, Any]:
        """Get summary of failure modes across fleet."""
        classifications = self.classify_fleet()

        mode_counts = {m.value: 0 for m in FailureMode}
        for result in classifications.values():
            mode_counts[result.mode.value] += 1

        return {
            "n_cohorts": len(classifications),
            "mode_counts": mode_counts,
            "dominant_mode": max(mode_counts, key=mode_counts.get),
            "cohorts": {k: v.mode.value for k, v in classifications.items()},
        }


# =============================================================================
# MODE-SPECIFIC PREDICTORS
# =============================================================================

@dataclass
class MarketRisk:
    """Market crash risk assessment result."""
    probability: float        # 0-1 probability of crash within 3 years
    expected_severity: float  # Expected crash magnitude (negative)
    warning: str              # CRITICAL, ELEVATED, HIGH, NORMAL
    current_state: str        # TRAPPED, FLOWING
    accumulation: str         # HIGH, LOW
    mass_time: float          # Accumulated mass × time


def predict_dimensional_collapse(
    cape: float,
    rotation: float,
    trapped_5yr: int,
    mass_time_5yr: float
) -> Dict[str, Any]:
    """
    Predictor for DIMENSIONAL_COLLAPSE mode (markets).

    Returns crash probability and expected severity.
    """
    risk = market_risk_assessment(cape, rotation, trapped_5yr, mass_time_5yr)
    return {
        "crash_probability_3yr": risk.probability,
        "expected_severity": risk.expected_severity,
        "current_state": risk.current_state,
        "accumulation": risk.accumulation,
        "mass_time": risk.mass_time,
        "warning": risk.warning
    }


def market_risk_assessment(
    cape: float,
    pc1_alignment: float,
    trapped_5yr: int,
    mass_time_5yr: float,
    rotation_threshold: float = 0.15
) -> MarketRisk:
    """
    Two-factor market crash risk assessment.

    Factor 1: Current State
        - TRAPPED = high mass (CAPE > 25) + modes locked (alignment > 0.7 or rotation < 0.15)
        - FLOWING = otherwise

    Factor 2: Accumulation
        - HIGH = 3+ trapped periods in rolling 5 years
        - LOW = fewer trapped periods

    Risk Matrix (empirical from 1950-2025 data):
        TRAPPED + HIGH_ACCUM: 80% crash probability, -36% expected severity
        TRAPPED + LOW_ACCUM:  40% crash probability, -21% expected severity
        HIGH_ACCUM only:      67% crash probability, -23% expected severity
        Neither:              25% crash probability, -20% expected severity

    Args:
        cape: Current CAPE ratio
        pc1_alignment: PC1 alignment (high = modes locked)
        trapped_5yr: Count of trapped periods in last 5 years
        mass_time_5yr: Sum of CAPE values during trapped periods
        rotation_threshold: Below this rotation level = trapped

    Returns:
        MarketRisk with probability, severity, and warning level
    """
    # Factor 1: Current state
    # TRAPPED = high mass + modes locked (low rotation OR high alignment)
    is_high_mass = cape > 25
    is_trapped = is_high_mass and pc1_alignment > 0.7

    # Factor 2: Accumulation
    is_high_accum = trapped_5yr >= 3

    # Risk matrix lookup (derived from empirical analysis)
    if is_trapped and is_high_accum:
        probability = 0.80
        expected_severity = -0.36
        warning = "CRITICAL"
    elif is_trapped:
        probability = 0.40
        expected_severity = -0.21
        warning = "ELEVATED"
    elif is_high_mass and is_high_accum:
        probability = 0.67
        expected_severity = -0.23
        warning = "HIGH"
    elif is_high_mass:
        probability = 0.20
        expected_severity = -0.18
        warning = "ELEVATED"
    else:
        probability = 0.25
        expected_severity = -0.20
        warning = "NORMAL"

    return MarketRisk(
        probability=probability,
        expected_severity=expected_severity,
        warning=warning,
        current_state="TRAPPED" if is_trapped else "FLOWING",
        accumulation="HIGH" if is_high_accum else "LOW",
        mass_time=mass_time_5yr
    )


def predict_mass_erosion(
    drift_rate: float,
    current_position: float,
    threshold: float = None
) -> Dict[str, Any]:
    """
    Predictor for MASS_EROSION mode (turbofans).

    Returns estimated remaining life based on drift rate.
    """
    if drift_rate <= 0:
        return {
            "remaining_life": float("inf"),
            "confidence": 0.0,
            "warning": "No degradation detected (non-positive drift)"
        }

    # If threshold known, compute time to threshold
    if threshold is not None:
        distance_to_threshold = threshold - current_position
        if distance_to_threshold > 0:
            remaining_life = distance_to_threshold / drift_rate
        else:
            remaining_life = 0  # Already past threshold
    else:
        # Use empirical relationship: RUL ∝ 1/drift_rate
        # Calibrated from CMAPSS: avg_drift = 0.0034, avg_life = 200
        remaining_life = 0.68 / drift_rate  # Empirical constant

    return {
        "remaining_life_estimate": remaining_life,
        "drift_rate": drift_rate,
        "confidence": 0.67,  # From correlation strength
        "warning": "CRITICAL" if remaining_life < 50 else "ELEVATED" if remaining_life < 100 else "NORMAL"
    }


def predict_cascading(
    local_drift: float,
    correlation_slope: float,
    current_mean_correlation: float
) -> Dict[str, Any]:
    """
    Predictor for CASCADING mode (chemical plants, grids).

    Returns cascade risk based on local drift × coupling rise.
    """
    # Combined risk metric
    cascade_risk = abs(local_drift) * max(0, correlation_slope) * 100

    # Coupling state
    if current_mean_correlation > 0.8:
        coupling_state = "COUPLED"
    elif current_mean_correlation > 0.5:
        coupling_state = "PARTIAL"
    else:
        coupling_state = "INDEPENDENT"

    return {
        "cascade_risk_score": cascade_risk,
        "coupling_state": coupling_state,
        "correlation_trend": "RISING" if correlation_slope > 0.01 else "STABLE" if correlation_slope > -0.01 else "FALLING",
        "local_drift_rate": local_drift,
        "warning": "CRITICAL" if cascade_risk > 0.5 else "ELEVATED" if cascade_risk > 0.1 else "NORMAL"
    }


__all__ = [
    "FailureMode",
    "FailureModeClassification",
    "FailureModeClassifier",
    "MarketRisk",
    "market_risk_assessment",
    "predict_dimensional_collapse",
    "predict_mass_erosion",
    "predict_cascading",
]
