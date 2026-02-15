"""
RUDDER Tuning Service
=====================

AI-guided tuning based on ground truth validation.

This service compares PRISM's detection results against ground truth labels
to learn optimal thresholds and identify which metrics work best for which
fault types.

Key capabilities:
    1. Load ground truth labels (from labels.parquet)
    2. Align PRISM detections to actual fault timestamps
    3. Compute lead times for each metric
    4. Learn fault signatures (metric → fault type mapping)
    5. Optimize detection thresholds
    6. Generate tuned configuration

Usage:
    from prime.services.tuning_service import TuningService

    tuner = TuningService("/path/to/data")
    result = tuner.tune()

    print(f"Optimal z-threshold: {result.optimal_z_threshold}")
    print(f"Best metrics: {result.best_metrics_by_fault_type}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

import duckdb
import polars as pl

from prime.sql.generate_readme import generate_sql_readme


@dataclass
class TuningResult:
    """Results from tuning against ground truth."""

    # Optimal threshold
    optimal_z_threshold: float
    optimal_z_criterion: str  # How optimal was selected

    # Best metrics by fault type
    best_metrics_by_fault_type: Dict[str, str]

    # Overall performance
    avg_lead_time: float
    detection_rate: float
    n_entities: int
    n_fault_types: int

    # Detailed results
    metric_rankings: List[Dict]
    threshold_curve: List[Dict]
    fault_signatures: List[Dict]

    # Human-readable recommendations
    recommendations: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "optimal_z_threshold": self.optimal_z_threshold,
            "optimal_z_criterion": self.optimal_z_criterion,
            "best_metrics_by_fault_type": self.best_metrics_by_fault_type,
            "avg_lead_time": self.avg_lead_time,
            "detection_rate": self.detection_rate,
            "n_entities": self.n_entities,
            "n_fault_types": self.n_fault_types,
            "metric_rankings": self.metric_rankings,
            "threshold_curve": self.threshold_curve,
            "fault_signatures": self.fault_signatures,
            "recommendations": self.recommendations,
        }


@dataclass
class TunedConfig:
    """Generated configuration based on tuning results."""

    z_warning: float
    z_critical: float
    priority_metrics: List[str]
    fault_signatures: Dict[str, str]
    tuning_metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tuned_thresholds": {
                "z_warning": self.z_warning,
                "z_critical": self.z_critical,
            },
            "priority_metrics": self.priority_metrics,
            "fault_signatures": self.fault_signatures,
            "tuning_metadata": self.tuning_metadata,
        }


class TuningService:
    """
    AI-guided tuning based on ground truth validation.

    The tuning workflow:
        1. Load physics.parquet (PRISM results)
        2. Load labels.parquet (ground truth)
        3. Align detection timestamps to fault timestamps
        4. Compute lead times (how early each metric detected each fault)
        5. Learn signatures (which metrics work best for which faults)
        6. Optimize thresholds (find z that maximizes detection rate)
        7. Generate tuned configuration
    """

    def __init__(self, data_dir: str, labels_path: Optional[str] = None):
        """
        Initialize tuning service.

        Args:
            data_dir: Directory containing PRISM results (physics.parquet, baseline_deviation.parquet)
            labels_path: Path to labels.parquet (defaults to data_dir/labels.parquet)
        """
        self.data_dir = Path(data_dir)
        self.labels_path = Path(labels_path) if labels_path else self.data_dir / "labels.parquet"
        self.conn = duckdb.connect()
        self._loaded = False
        self._sql_dir = Path(__file__).parent.parent / "sql"

    def _load_data(self):
        """Load observations, physics results, and labels."""
        if self._loaded:
            return

        # Load physics.parquet
        physics_path = self.data_dir / "physics.parquet"
        if physics_path.exists():
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS physics AS SELECT * FROM read_parquet('{physics_path}')")
        else:
            raise FileNotFoundError(f"Physics results not found: {physics_path}")

        # Load baseline_deviation.parquet (for z_total)
        baseline_path = self.data_dir / "baseline_deviation.parquet"
        if baseline_path.exists():
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS baseline_deviation AS SELECT * FROM read_parquet('{baseline_path}')")
        else:
            # Try alternate name
            baseline_path = self.data_dir / "deviation.parquet"
            if baseline_path.exists():
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS baseline_deviation AS SELECT * FROM read_parquet('{baseline_path}')")

        # Load labels.parquet
        if self.labels_path.exists():
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS labels AS SELECT * FROM read_parquet('{self.labels_path}')")
        else:
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        # Run ground truth SQL files
        self._run_sql_file("60_ground_truth.sql")
        self._run_sql_file("61_lead_time_analysis.sql")
        self._run_sql_file("62_fault_signatures.sql")
        self._run_sql_file("63_threshold_optimization.sql")

        # Generate SQL README alongside output
        try:
            generate_sql_readme(self.conn, self.data_dir)
        except Exception:
            pass  # README generation is non-critical

        self._loaded = True

    def _run_sql_file(self, filename: str):
        """Execute a SQL file."""
        sql_path = self._sql_dir / filename
        if not sql_path.exists():
            print(f"Warning: SQL file not found: {sql_path}")
            return

        sql_content = sql_path.read_text()

        # Execute each statement
        for statement in sql_content.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    self.conn.execute(statement)
                except Exception as e:
                    # Skip errors (view might already exist, etc.)
                    pass

    def get_label_summary(self) -> pl.DataFrame:
        """Get summary of available labels."""
        self._load_data()
        try:
            return self.conn.execute("SELECT * FROM v_label_summary").pl()
        except Exception:
            return pl.DataFrame()

    def analyze_lead_times(self) -> pl.DataFrame:
        """Compute lead time for each metric against ground truth."""
        self._load_data()
        try:
            return self.conn.execute("SELECT * FROM v_metric_lead_times").pl()
        except Exception:
            return pl.DataFrame()

    def get_metric_performance(self) -> pl.DataFrame:
        """Get aggregated metric performance (which metrics detect best)."""
        self._load_data()
        try:
            return self.conn.execute("SELECT * FROM v_metric_performance").pl()
        except Exception:
            return pl.DataFrame()

    def get_threshold_curve(self) -> pl.DataFrame:
        """Get threshold performance curve data."""
        self._load_data()
        try:
            return self.conn.execute("SELECT * FROM v_threshold_curve").pl()
        except Exception:
            return pl.DataFrame()

    def find_optimal_threshold(self) -> Dict:
        """Find z-threshold that maximizes early detection rate."""
        self._load_data()
        try:
            df = self.conn.execute("SELECT * FROM v_optimal_threshold").pl()
            if df.height == 0:
                return {"optimal_z": 2.5, "criterion": "default", "detection_rate": 0, "lead_time": 0}

            # Get the "best_balanced" criterion
            best = df.filter(pl.col("criterion") == "best_balanced")
            if best.height == 0:
                best = df.head(1)

            row = best.to_dicts()[0]
            return {
                "optimal_z": row.get("optimal_z", 2.5),
                "criterion": row.get("criterion", "unknown"),
                "detection_rate": row.get("detection_rate_pct", 0),
                "lead_time": row.get("avg_lead_time", 0),
            }
        except Exception as e:
            print(f"Error finding optimal threshold: {e}")
            return {"optimal_z": 2.5, "criterion": "default", "detection_rate": 0, "lead_time": 0}

    def learn_fault_signatures(self) -> Dict[str, str]:
        """Learn which metrics detect which fault types best."""
        self._load_data()
        try:
            df = self.conn.execute("SELECT * FROM v_best_detectors").pl()
            if df.height == 0:
                return {}

            signatures = {}
            for row in df.iter_rows(named=True):
                fault_type = row.get("fault_type", "unknown")
                best_metric = row.get("best_metric", "coherence")
                signatures[fault_type] = best_metric

            return signatures
        except Exception as e:
            print(f"Error learning signatures: {e}")
            return {}

    def get_fault_signature_matrix(self) -> pl.DataFrame:
        """Get the fault type × metric performance matrix."""
        self._load_data()
        try:
            return self.conn.execute("SELECT * FROM v_fault_signature_matrix").pl()
        except Exception:
            return pl.DataFrame()

    def get_detection_summary(self) -> Dict:
        """Get overall detection vs ground truth summary."""
        self._load_data()
        try:
            df = self.conn.execute("SELECT * FROM v_detection_outcome_summary").pl()
            if df.height == 0:
                return {"n_entities": 0, "detection_rate": 0, "avg_lead_time": 0}

            # Aggregate across all label types
            return {
                "n_entities": df["n_entities"].sum(),
                "early_detections": df["early_2sigma"].sum(),
                "missed": df["missed_2sigma"].sum(),
                "detection_rate": round(df["detection_rate_2sigma"].mean(), 1),
                "avg_lead_time": round(df["avg_lead_time_2sigma"].mean(), 1) if df["avg_lead_time_2sigma"].is_not_null().any() else 0,
            }
        except Exception as e:
            print(f"Error getting detection summary: {e}")
            return {"n_entities": 0, "detection_rate": 0, "avg_lead_time": 0}

    def generate_tuned_config(self) -> TunedConfig:
        """Generate optimized manifest based on tuning results."""
        signatures = self.learn_fault_signatures()
        threshold_info = self.find_optimal_threshold()

        optimal_z = threshold_info.get("optimal_z", 2.5)

        # Get unique metrics from signatures
        priority_metrics = list(set(signatures.values())) if signatures else ["coherence", "entropy"]

        return TunedConfig(
            z_warning=optimal_z,
            z_critical=optimal_z + 1.0,
            priority_metrics=priority_metrics,
            fault_signatures=signatures,
            tuning_metadata={
                "tuning_criterion": threshold_info.get("criterion", "unknown"),
                "detection_rate": threshold_info.get("detection_rate", 0),
                "avg_lead_time": threshold_info.get("lead_time", 0),
                "n_fault_types": len(signatures),
            }
        )

    def _generate_recommendations(self, detection_summary: Dict, threshold_info: Dict, signatures: Dict) -> str:
        """Generate human-readable recommendations."""
        lines = ["## Tuning Recommendations\n"]

        # Threshold recommendation
        optimal_z = threshold_info.get("optimal_z", 2.5)
        detection_rate = threshold_info.get("detection_rate", 0)
        lines.append(f"### Threshold")
        lines.append(f"- Use **z = {optimal_z}** for warnings (detection rate: {detection_rate}%)")
        lines.append(f"- Use **z = {optimal_z + 1.0}** for critical alerts")
        lines.append("")

        # Priority metrics
        if signatures:
            lines.append("### Priority Metrics by Fault Type")
            for fault_type, metric in signatures.items():
                lines.append(f"- **{fault_type}**: {metric}")
            lines.append("")

        # Lead time
        avg_lead = detection_summary.get("avg_lead_time", 0)
        if avg_lead > 0:
            lines.append("### Lead Time")
            lines.append(f"- Average early warning: **{avg_lead:.1f} samples** before fault")
            lines.append("")

        # Detection performance
        detection_rate_overall = detection_summary.get("detection_rate", 0)
        n_entities = detection_summary.get("n_entities", 0)
        lines.append("### Detection Performance")
        lines.append(f"- Overall detection rate: **{detection_rate_overall}%**")
        lines.append(f"- Entities analyzed: **{n_entities}**")

        return "\n".join(lines)

    def tune(self) -> TuningResult:
        """Run full tuning analysis."""
        self._load_data()

        # Get all analysis results
        detection_summary = self.get_detection_summary()
        threshold_info = self.find_optimal_threshold()
        signatures = self.learn_fault_signatures()
        metric_perf = self.get_metric_performance()
        threshold_curve = self.get_threshold_curve()
        signature_matrix = self.get_fault_signature_matrix()

        # Generate recommendations
        recommendations = self._generate_recommendations(
            detection_summary, threshold_info, signatures
        )

        return TuningResult(
            optimal_z_threshold=threshold_info.get("optimal_z", 2.5),
            optimal_z_criterion=threshold_info.get("criterion", "unknown"),
            best_metrics_by_fault_type=signatures,
            avg_lead_time=detection_summary.get("avg_lead_time", 0),
            detection_rate=detection_summary.get("detection_rate", 0),
            n_entities=detection_summary.get("n_entities", 0),
            n_fault_types=len(signatures),
            metric_rankings=metric_perf.to_dicts() if metric_perf.height > 0 else [],
            threshold_curve=threshold_curve.to_dicts() if threshold_curve.height > 0 else [],
            fault_signatures=signature_matrix.to_dicts() if signature_matrix.height > 0 else [],
            recommendations=recommendations,
        )


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_tuning_service: Optional[TuningService] = None


def get_tuning_service(data_dir: str = None, labels_path: str = None) -> TuningService:
    """Get or create tuning service singleton."""
    global _tuning_service

    if data_dir is not None:
        # Create new service for this data directory
        _tuning_service = TuningService(data_dir, labels_path)

    if _tuning_service is None:
        raise ValueError("TuningService not initialized. Provide data_dir.")

    return _tuning_service
