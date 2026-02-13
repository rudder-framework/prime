"""
RUDDER Fingerprint Service
==========================

Manage system health fingerprints - the learned signatures of healthy,
deviation, and failure states.

Same math. Different fingerprints. Every system has its own signature.

Three fingerprint types:
    HEALTHY   - "Normal looks like this" (baseline reference)
    DEVIATION - "Trouble starting looks like this" (early warning)
    FAILURE   - "System has failed" (post-mortem learning)

Usage:
    from framework.services.fingerprint_service import FingerprintService

    fps = FingerprintService()

    # Load existing fingerprints
    fps.load_fingerprints("pump")

    # Compare current state to fingerprints
    match = fps.match_deviation(current_metrics)
    print(f"Matches {match.fault_type} with {match.confidence}% confidence")
    print(f"Expected lead time: {match.lead_time} samples")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import yaml
import json

import polars as pl


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MetricRange:
    """Range and statistics for a metric in a fingerprint."""
    mean: float
    std: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"mean": self.mean, "std": self.std}
        if self.min_val is not None:
            d["range"] = [self.min_val, self.max_val]
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "MetricRange":
        range_vals = d.get("range", [None, None])
        return cls(
            mean=d.get("mean", 0),
            std=d.get("std", 0),
            min_val=range_vals[0] if range_vals else None,
            max_val=range_vals[1] if len(range_vals) > 1 else None,
            note=d.get("note"),
        )


@dataclass
class DeviationIndicator:
    """An indicator of deviation from healthy baseline."""
    metric: str
    direction: str  # "increasing", "decreasing", "crossing"
    trigger: str  # e.g., "z < -2.0" or "crosses 0.5"
    typical_lead_time: int  # samples before fault
    delta_from_baseline: Optional[float] = None
    z_score_trigger: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "trigger": self.trigger,
            "typical_lead_time": self.typical_lead_time,
            "delta_from_baseline": self.delta_from_baseline,
            "z_score_trigger": self.z_score_trigger,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DeviationIndicator":
        return cls(
            metric=d.get("metric", ""),
            direction=d.get("direction", ""),
            trigger=d.get("trigger", ""),
            typical_lead_time=d.get("typical_lead_time", 0),
            delta_from_baseline=d.get("delta_from_baseline"),
            z_score_trigger=d.get("z_score_trigger"),
            note=d.get("note"),
        )


@dataclass
class HealthyFingerprint:
    """Fingerprint of a healthy system - the baseline reference."""
    system_id: str
    domain: str
    captured_from: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Metric categories
    geometry: Dict[str, MetricRange] = field(default_factory=dict)
    dynamics: Dict[str, MetricRange] = field(default_factory=dict)
    information: Dict[str, MetricRange] = field(default_factory=dict)
    primitives: Dict[str, MetricRange] = field(default_factory=dict)

    # Summary
    signature_summary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "fingerprint_type": "healthy",
            "system_id": self.system_id,
            "domain": self.domain,
            "captured_from": self.captured_from,
            "created_at": self.created_at,
            "geometry": {k: v.to_dict() for k, v in self.geometry.items()},
            "dynamics": {k: v.to_dict() for k, v in self.dynamics.items()},
            "information": {k: v.to_dict() for k, v in self.information.items()},
            "primitives": {k: v.to_dict() for k, v in self.primitives.items()},
            "signature_summary": self.signature_summary,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HealthyFingerprint":
        return cls(
            system_id=d.get("system_id", ""),
            domain=d.get("domain", ""),
            captured_from=d.get("captured_from", ""),
            created_at=d.get("created_at", ""),
            geometry={k: MetricRange.from_dict(v) for k, v in d.get("geometry", {}).items()},
            dynamics={k: MetricRange.from_dict(v) for k, v in d.get("dynamics", {}).items()},
            information={k: MetricRange.from_dict(v) for k, v in d.get("information", {}).items()},
            primitives={k: MetricRange.from_dict(v) for k, v in d.get("primitives", {}).items()},
            signature_summary=d.get("signature_summary", []),
        )


@dataclass
class DeviationFingerprint:
    """Fingerprint of a deviation pattern - the early warning signature."""
    system_id: str
    domain: str
    fault_type: str
    captured_from: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Primary and secondary indicators
    primary_indicators: List[DeviationIndicator] = field(default_factory=list)
    secondary_indicators: List[DeviationIndicator] = field(default_factory=list)

    # Pattern description
    pattern_description: str = ""

    # Detection rule
    detection_condition: str = ""  # e.g., "coherence_z < -2.0 AND lyapunov_delta > 0.02"
    confidence: float = 0.0
    expected_lead_time: int = 0
    false_positive_rate: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "fingerprint_type": "deviation",
            "system_id": self.system_id,
            "domain": self.domain,
            "fault_type": self.fault_type,
            "captured_from": self.captured_from,
            "created_at": self.created_at,
            "deviation_signature": {
                "primary_indicators": [i.to_dict() for i in self.primary_indicators],
                "secondary_indicators": [i.to_dict() for i in self.secondary_indicators],
                "pattern_description": self.pattern_description,
            },
            "detection_rule": {
                "condition": self.detection_condition,
                "confidence": self.confidence,
                "expected_lead_time": self.expected_lead_time,
                "false_positive_rate": self.false_positive_rate,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DeviationFingerprint":
        sig = d.get("deviation_signature", {})
        rule = d.get("detection_rule", {})
        return cls(
            system_id=d.get("system_id", ""),
            domain=d.get("domain", ""),
            fault_type=d.get("fault_type", ""),
            captured_from=d.get("captured_from", ""),
            created_at=d.get("created_at", ""),
            primary_indicators=[DeviationIndicator.from_dict(i) for i in sig.get("primary_indicators", [])],
            secondary_indicators=[DeviationIndicator.from_dict(i) for i in sig.get("secondary_indicators", [])],
            pattern_description=sig.get("pattern_description", ""),
            detection_condition=rule.get("condition", ""),
            confidence=rule.get("confidence", 0),
            expected_lead_time=rule.get("expected_lead_time", 0),
            false_positive_rate=rule.get("false_positive_rate", 0),
        )


@dataclass
class FailureFingerprint:
    """Fingerprint of a failure state - what full failure looks like."""
    system_id: str
    domain: str
    fault_type: str
    captured_from: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Terminal state values (metric -> value, vs_healthy_pct)
    terminal_values: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Definitive markers
    definitive_markers: List[str] = field(default_factory=list)

    # Pattern description
    pattern_description: str = ""

    # Too late indicators
    too_late_indicators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "fingerprint_type": "failure",
            "system_id": self.system_id,
            "domain": self.domain,
            "fault_type": self.fault_type,
            "captured_from": self.captured_from,
            "created_at": self.created_at,
            "terminal_values": self.terminal_values,
            "failure_signature": {
                "definitive_markers": self.definitive_markers,
                "pattern_description": self.pattern_description,
            },
            "too_late_indicators": self.too_late_indicators,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FailureFingerprint":
        sig = d.get("failure_signature", {})
        return cls(
            system_id=d.get("system_id", ""),
            domain=d.get("domain", ""),
            fault_type=d.get("fault_type", ""),
            captured_from=d.get("captured_from", ""),
            created_at=d.get("created_at", ""),
            terminal_values=d.get("terminal_values", {}),
            definitive_markers=sig.get("definitive_markers", []),
            pattern_description=sig.get("pattern_description", ""),
            too_late_indicators=d.get("too_late_indicators", []),
        )


@dataclass
class FingerprintMatch:
    """Result of matching current state to a fingerprint."""
    fingerprint_type: str  # "healthy", "deviation", "failure"
    fault_type: Optional[str]
    confidence: float
    lead_time: Optional[int]
    matching_indicators: List[str]
    pattern_description: str


# =============================================================================
# FINGERPRINT SERVICE
# =============================================================================

class FingerprintService:
    """
    Manage and match system health fingerprints.

    Workflow:
    1. Load fingerprints for a domain
    2. Compare current metrics to healthy baseline
    3. If deviating, match to known deviation patterns
    4. Return match with confidence and lead time
    """

    def __init__(self, fingerprints_dir: Optional[str] = None):
        """
        Initialize fingerprint service.

        Args:
            fingerprints_dir: Directory containing fingerprint YAML files.
                             Defaults to ~/.rudder/fingerprints/
        """
        if fingerprints_dir:
            self.fingerprints_dir = Path(fingerprints_dir)
        else:
            self.fingerprints_dir = Path.home() / ".rudder" / "fingerprints"

        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)

        # Loaded fingerprints
        self.healthy: Dict[str, HealthyFingerprint] = {}
        self.deviations: Dict[str, List[DeviationFingerprint]] = {}
        self.failures: Dict[str, List[FailureFingerprint]] = {}

    def load_fingerprints(self, domain: str) -> int:
        """
        Load all fingerprints for a domain.

        Args:
            domain: Domain name (e.g., "pump", "turbofan", "bearing")

        Returns:
            Number of fingerprints loaded
        """
        domain_dir = self.fingerprints_dir / domain
        if not domain_dir.exists():
            return 0

        count = 0

        for fp_file in domain_dir.glob("*.yaml"):
            try:
                with open(fp_file) as f:
                    data = yaml.safe_load(f)

                fp_type = data.get("fingerprint_type")

                if fp_type == "healthy":
                    fp = HealthyFingerprint.from_dict(data)
                    self.healthy[domain] = fp
                    count += 1

                elif fp_type == "deviation":
                    fp = DeviationFingerprint.from_dict(data)
                    if domain not in self.deviations:
                        self.deviations[domain] = []
                    self.deviations[domain].append(fp)
                    count += 1

                elif fp_type == "failure":
                    fp = FailureFingerprint.from_dict(data)
                    if domain not in self.failures:
                        self.failures[domain] = []
                    self.failures[domain].append(fp)
                    count += 1

            except Exception as e:
                print(f"Warning: Failed to load {fp_file}: {e}")

        return count

    def save_fingerprint(self, fingerprint, domain: str, filename: str):
        """Save a fingerprint to YAML file."""
        domain_dir = self.fingerprints_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        fp_path = domain_dir / f"{filename}.yaml"
        with open(fp_path, 'w') as f:
            yaml.dump(fingerprint.to_dict(), f, default_flow_style=False, sort_keys=False)

        return fp_path

    def get_healthy_baseline(self, domain: str) -> Optional[HealthyFingerprint]:
        """Get the healthy baseline fingerprint for a domain."""
        return self.healthy.get(domain)

    def get_deviation_fingerprints(self, domain: str) -> List[DeviationFingerprint]:
        """Get all deviation fingerprints for a domain."""
        return self.deviations.get(domain, [])

    def compare_to_healthy(
        self,
        metrics: Dict[str, float],
        domain: str
    ) -> Dict[str, Tuple[float, str]]:
        """
        Compare current metrics to healthy baseline.

        Args:
            metrics: Current metric values (e.g., {"coherence": 0.18, "lyapunov": -0.03})
            domain: Domain to compare against

        Returns:
            Dict of metric -> (z_score, status) where status is "normal", "warning", "critical"
        """
        baseline = self.healthy.get(domain)
        if not baseline:
            return {}

        results = {}

        # Check each category
        for category_name, category in [
            ("geometry", baseline.geometry),
            ("dynamics", baseline.dynamics),
            ("information", baseline.information),
            ("primitives", baseline.primitives),
        ]:
            for metric_name, metric_range in category.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if metric_range.std > 0:
                        z_score = (value - metric_range.mean) / metric_range.std
                    else:
                        z_score = 0.0

                    if abs(z_score) < 2.0:
                        status = "normal"
                    elif abs(z_score) < 3.0:
                        status = "warning"
                    else:
                        status = "critical"

                    results[metric_name] = (z_score, status)

        return results

    def match_deviation(
        self,
        metrics: Dict[str, float],
        domain: str,
        z_scores: Optional[Dict[str, float]] = None
    ) -> Optional[FingerprintMatch]:
        """
        Match current metrics to a deviation fingerprint.

        Args:
            metrics: Current metric values
            domain: Domain to match against
            z_scores: Pre-computed z-scores (if available)

        Returns:
            FingerprintMatch if a pattern matches, None otherwise
        """
        deviation_fps = self.deviations.get(domain, [])
        if not deviation_fps:
            return None

        # Compute z-scores if not provided
        if z_scores is None:
            comparison = self.compare_to_healthy(metrics, domain)
            z_scores = {k: v[0] for k, v in comparison.items()}

        best_match = None
        best_score = 0.0

        for fp in deviation_fps:
            matching_primary = []
            matching_secondary = []
            score = 0.0

            # Check primary indicators
            for indicator in fp.primary_indicators:
                metric = indicator.metric
                if metric in z_scores:
                    z = z_scores[metric]

                    # Parse trigger condition
                    if self._check_trigger(z, metrics.get(metric), indicator.trigger):
                        matching_primary.append(indicator.metric)
                        score += 2.0  # Primary indicators worth more

            # Check secondary indicators
            for indicator in fp.secondary_indicators:
                metric = indicator.metric
                if metric in z_scores:
                    z = z_scores[metric]

                    if self._check_trigger(z, metrics.get(metric), indicator.trigger):
                        matching_secondary.append(indicator.metric)
                        score += 1.0

            # Compute confidence based on matches
            total_indicators = len(fp.primary_indicators) + len(fp.secondary_indicators)
            if total_indicators > 0:
                confidence = (
                    len(matching_primary) * 2 + len(matching_secondary)
                ) / (len(fp.primary_indicators) * 2 + len(fp.secondary_indicators)) * 100

                # Must have at least one primary indicator matching
                if len(matching_primary) > 0 and score > best_score:
                    best_score = score
                    best_match = FingerprintMatch(
                        fingerprint_type="deviation",
                        fault_type=fp.fault_type,
                        confidence=confidence,
                        lead_time=fp.expected_lead_time,
                        matching_indicators=matching_primary + matching_secondary,
                        pattern_description=fp.pattern_description,
                    )

        return best_match

    def _check_trigger(self, z_score: float, value: Optional[float], trigger: str) -> bool:
        """Check if a trigger condition is met."""
        trigger_lower = trigger.lower().strip()

        # Handle z-score based triggers: "z < -2.0", "z > 2.5"
        if "z" in trigger_lower:
            import re
            # Match patterns like "z < -2.0", "z > 2.5", "z < 2.0"
            match = re.search(r'z\s*([<>])\s*(-?\d+\.?\d*)', trigger_lower)
            if match:
                op = match.group(1)
                threshold = float(match.group(2))
                if op == '<':
                    return z_score < threshold
                elif op == '>':
                    return z_score > threshold

        # Handle value-based triggers: "crosses 0.5"
        if value is not None:
            if "crosses" in trigger_lower:
                try:
                    threshold = float(trigger_lower.split()[-1])
                    # "crosses 0.5" means value has moved from below to above (or near) threshold
                    return value > threshold or abs(value - threshold) < 0.05
                except ValueError:
                    pass

        return False

    def check_failure(
        self,
        metrics: Dict[str, float],
        domain: str
    ) -> Optional[FingerprintMatch]:
        """
        Check if metrics indicate system failure.

        Args:
            metrics: Current metric values
            domain: Domain to check against

        Returns:
            FingerprintMatch if in failure state, None otherwise
        """
        failure_fps = self.failures.get(domain, [])
        if not failure_fps:
            return None

        for fp in failure_fps:
            matching_markers = []

            for marker in fp.definitive_markers:
                # Parse marker (e.g., "coherence < 0.12")
                try:
                    parts = marker.replace("<", " < ").replace(">", " > ").split()
                    metric = parts[0]
                    op = parts[1]
                    threshold = float(parts[2])

                    if metric in metrics:
                        value = metrics[metric]
                        if op == "<" and value < threshold:
                            matching_markers.append(marker)
                        elif op == ">" and value > threshold:
                            matching_markers.append(marker)
                except Exception:
                    continue

            # If majority of markers match, it's a failure
            if len(matching_markers) >= len(fp.definitive_markers) * 0.6:
                return FingerprintMatch(
                    fingerprint_type="failure",
                    fault_type=fp.fault_type,
                    confidence=len(matching_markers) / len(fp.definitive_markers) * 100,
                    lead_time=0,  # Too late
                    matching_indicators=matching_markers,
                    pattern_description=fp.pattern_description,
                )

        return None

    def classify_state(
        self,
        metrics: Dict[str, float],
        domain: str
    ) -> FingerprintMatch:
        """
        Classify current system state against all fingerprints.

        Returns the best matching fingerprint (healthy, deviation, or failure).

        Args:
            metrics: Current metric values
            domain: Domain to classify against

        Returns:
            FingerprintMatch with the current state classification
        """
        # First check for failure (if in failure, skip deviation)
        failure_match = self.check_failure(metrics, domain)
        if failure_match and failure_match.confidence > 60:
            return failure_match

        # Check for deviation
        deviation_match = self.match_deviation(metrics, domain)
        if deviation_match and deviation_match.confidence > 50:
            return deviation_match

        # Otherwise, compare to healthy
        comparison = self.compare_to_healthy(metrics, domain)
        anomaly_count = sum(1 for _, (z, status) in comparison.items() if status != "normal")

        if anomaly_count == 0:
            return FingerprintMatch(
                fingerprint_type="healthy",
                fault_type=None,
                confidence=100.0,
                lead_time=None,
                matching_indicators=[],
                pattern_description="System operating within normal parameters",
            )
        else:
            # Some deviation but no matching pattern
            return FingerprintMatch(
                fingerprint_type="deviation",
                fault_type="unknown",
                confidence=30.0,  # Low confidence without pattern match
                lead_time=None,
                matching_indicators=[m for m, (z, s) in comparison.items() if s != "normal"],
                pattern_description="Metrics deviating from baseline but no known pattern match",
            )


# =============================================================================
# FINGERPRINT GENERATOR (from tuning results)
# =============================================================================

def generate_healthy_fingerprint(
    tuning_service,
    system_id: str,
    domain: str,
    baseline_description: str = ""
) -> HealthyFingerprint:
    """
    Generate a healthy fingerprint from tuning service baseline data.

    Args:
        tuning_service: TuningService instance with loaded data
        system_id: System identifier
        domain: Domain name
        baseline_description: Description of baseline period

    Returns:
        HealthyFingerprint populated from baseline statistics
    """
    # Get baseline stats from tuning service
    baseline_df = tuning_service.conn.execute("""
        SELECT
            metric_name,
            AVG(baseline_mean) as mean,
            AVG(baseline_std) as std,
            MIN(baseline_min) as min_val,
            MAX(baseline_max) as max_val
        FROM v_metric_baseline
        GROUP BY metric_name
    """).pl()

    geometry = {}
    dynamics = {}
    information = {}

    for row in baseline_df.iter_rows(named=True):
        metric = row["metric_name"]
        mr = MetricRange(
            mean=row["mean"] or 0,
            std=row["std"] or 0,
            min_val=row["min_val"],
            max_val=row["max_val"],
        )

        # Categorize metrics
        if metric in ["coherence", "effective_dimension", "mean_correlation"]:
            geometry[metric] = mr
        elif metric in ["lyapunov", "rqa_determinism", "stability"]:
            dynamics[metric] = mr
        elif metric in ["entropy", "hurst", "transfer_entropy"]:
            information[metric] = mr

    return HealthyFingerprint(
        system_id=system_id,
        domain=domain,
        captured_from=baseline_description,
        geometry=geometry,
        dynamics=dynamics,
        information=information,
        signature_summary=[],
    )


def generate_deviation_fingerprint(
    tuning_service,
    system_id: str,
    domain: str,
    fault_type: str,
    captured_description: str = ""
) -> DeviationFingerprint:
    """
    Generate a deviation fingerprint from tuning results.

    Args:
        tuning_service: TuningService with loaded tuning results
        system_id: System identifier
        domain: Domain name
        fault_type: Fault type label
        captured_description: Description of data source

    Returns:
        DeviationFingerprint populated from tuning analysis
    """
    # Get metric performance for this fault type
    perf_df = tuning_service.conn.execute(f"""
        SELECT *
        FROM v_metric_performance
        WHERE label_name = '{fault_type}'
        ORDER BY avg_lead_time DESC NULLS LAST
    """).pl()

    primary = []
    secondary = []

    for i, row in enumerate(perf_df.iter_rows(named=True)):
        if row["detection_rate_pct"] is None:
            continue

        indicator = DeviationIndicator(
            metric=row["metric_name"],
            direction="deviating",
            trigger=f"z > 2.0",
            typical_lead_time=int(row["avg_lead_time"]) if row["avg_lead_time"] else 0,
            z_score_trigger=2.0,
        )

        if i < 2 and row["detection_rate_pct"] > 60:
            primary.append(indicator)
        elif row["detection_rate_pct"] > 40:
            secondary.append(indicator)

    # Get expected lead time
    lead_time = int(perf_df["avg_lead_time"].mean()) if perf_df.height > 0 else 0
    detection_rate = perf_df["detection_rate_pct"].mean() if perf_df.height > 0 else 0

    return DeviationFingerprint(
        system_id=system_id,
        domain=domain,
        fault_type=fault_type,
        captured_from=captured_description,
        primary_indicators=primary,
        secondary_indicators=secondary,
        detection_condition="",
        confidence=detection_rate or 0,
        expected_lead_time=lead_time,
        false_positive_rate=100 - (detection_rate or 0),
    )


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_fingerprint_service: Optional[FingerprintService] = None


def get_fingerprint_service(fingerprints_dir: str = None) -> FingerprintService:
    """Get or create fingerprint service singleton."""
    global _fingerprint_service

    if fingerprints_dir is not None or _fingerprint_service is None:
        _fingerprint_service = FingerprintService(fingerprints_dir)

    return _fingerprint_service
