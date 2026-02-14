"""
State Distance Analyzer
=======================

Interprets PRISM's state.parquet output.
Detects anomalies, transitions, and trends.

Key insight: state_velocity is the generalized hd_slope.
- hd_slope only tracked hurst
- state_velocity tracks ALL metrics

state.parquet schema:
    entity_id          : Utf8     - Which entity
    I                  : Float64  - Index (time, cycle)
    state_distance     : Float64  - Mahalanobis distance from baseline
    state_velocity     : Float64  - d(state_distance)/dI
    state_acceleration : Float64  - d²(state_distance)/dI²
    n_metrics_used     : Int32    - How many metrics contributed

Interpretation:
    state_distance:
        = 0     → Exactly at baseline
        = 1     → 1σ from baseline (normal variation)
        = 2     → 2σ from baseline (notable)
        = 3+    → Significantly different (investigate)

    state_velocity:
        ≈ 0     → Stable (good)
        > 0     → Moving AWAY from baseline (degrading?)
        < 0     → Moving TOWARD baseline (recovering?)

    state_acceleration:
        > 0     → Degradation speeding up (bad)
        < 0     → Degradation slowing (stabilizing)
        ≈ 0     → Constant rate of change
"""

import polars as pl
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict


@dataclass
class StateThresholds:
    """Configurable thresholds for state interpretation."""
    distance_warning: float = 2.0      # σ from baseline
    distance_critical: float = 3.0     # σ from baseline
    velocity_warning: float = 0.05     # units per I
    velocity_critical: float = 0.1     # units per I
    acceleration_warning: float = 0.01 # acceleration threshold

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class StateAnalyzer:
    """
    Analyzes state distance output from PRISM.

    Key insight: state_velocity is the generalized hd_slope.
    - hd_slope only tracked hurst
    - state_velocity tracks ALL metrics
    """

    def __init__(
        self,
        state_path: Union[str, Path] = None,
        state_df: pl.DataFrame = None,
        thresholds: StateThresholds = None
    ):
        """
        Initialize analyzer.

        Args:
            state_path: Path to state.parquet file
            state_df: Or provide DataFrame directly
            thresholds: Custom thresholds (optional)
        """
        if state_df is not None:
            self.state = state_df
        elif state_path:
            self.state = pl.read_parquet(state_path)
        else:
            raise ValueError("Must provide state_path or state_df")

        self.thresholds = thresholds or StateThresholds()

    def get_entities(self) -> List[str]:
        """Get list of all entities in state data."""
        return self.state.select("entity_id").unique().to_series().to_list()

    def get_current_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get most recent state for an entity."""
        latest = (
            self.state
            .filter(pl.col("entity_id") == entity_id)
            .sort("I", descending=True)
            .head(1)
            .to_dicts()
        )

        if not latest:
            return None

        row = latest[0]

        # Interpret
        status = self._interpret_state(
            row.get('state_distance', 0),
            row.get('state_velocity', 0),
            row.get('state_acceleration', 0)
        )

        return {
            **row,
            'status': status,
            'status_label': self._status_label(status),
        }

    def _interpret_state(
        self,
        distance: float,
        velocity: float,
        acceleration: float
    ) -> str:
        """Interpret state into status category."""
        t = self.thresholds

        # Handle None values
        distance = distance or 0
        velocity = velocity or 0
        acceleration = acceleration or 0

        # Critical: far from baseline AND moving away fast
        if distance > t.distance_critical and velocity > t.velocity_critical:
            return 'critical'

        # Warning: elevated distance OR significant velocity
        if distance > t.distance_warning or velocity > t.velocity_warning:
            return 'warning'

        # Watch: accelerating degradation
        if acceleration > t.acceleration_warning and velocity > 0:
            return 'watch'

        return 'normal'

    def _status_label(self, status: str) -> str:
        """Human-readable status label."""
        return {
            'normal': 'Normal Operation',
            'watch': 'Monitoring - Acceleration Detected',
            'warning': 'Warning - Elevated State Distance',
            'critical': 'Critical - Significant Deviation',
        }.get(status, 'Unknown')

    def get_state_trajectory(
        self,
        entity_id: str,
        start_I: float = None,
        end_I: float = None
    ) -> pl.DataFrame:
        """Get state trajectory for an entity."""
        query = self.state.filter(pl.col("entity_id") == entity_id)

        if start_I is not None:
            query = query.filter(pl.col("I") >= start_I)
        if end_I is not None:
            query = query.filter(pl.col("I") <= end_I)

        return query.sort("I")

    def find_transitions(
        self,
        entity_id: str,
        velocity_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Find points where state velocity crosses threshold.
        These are transition events - system starting to degrade or recover.
        """
        if velocity_threshold is None:
            velocity_threshold = self.thresholds.velocity_warning

        df = self.get_state_trajectory(entity_id)

        if df.is_empty():
            return []

        # Add previous velocity
        df = df.with_columns(
            pl.col("state_velocity").shift(1).alias("prev_velocity")
        )

        # Find crossings
        transitions = df.filter(
            # Crossed above threshold (started degrading)
            ((pl.col("prev_velocity") <= velocity_threshold) &
             (pl.col("state_velocity") > velocity_threshold)) |
            # Crossed below negative threshold (started recovering)
            ((pl.col("prev_velocity") >= -velocity_threshold) &
             (pl.col("state_velocity") < -velocity_threshold))
        ).to_dicts()

        for t in transitions:
            vel = t.get('state_velocity', 0) or 0
            t['transition_type'] = 'degradation_start' if vel > 0 else 'recovery_start'

        return transitions

    def find_anomalies(
        self,
        entity_id: str = None,
        distance_threshold: float = None
    ) -> pl.DataFrame:
        """Find all points exceeding distance threshold."""
        if distance_threshold is None:
            distance_threshold = self.thresholds.distance_critical

        query = self.state.filter(pl.col("state_distance") > distance_threshold)

        if entity_id:
            query = query.filter(pl.col("entity_id") == entity_id)

        return query.sort("entity_id", "I")

    def get_all_current_states(self) -> List[Dict[str, Any]]:
        """Get current state for all entities."""
        # Get latest I for each entity
        latest = (
            self.state
            .group_by("entity_id")
            .agg(pl.col("I").max().alias("max_I"))
        )

        # Join to get latest rows
        current = (
            self.state
            .join(latest, on="entity_id")
            .filter(pl.col("I") == pl.col("max_I"))
            .drop("max_I")
        )

        results = []
        for row in current.to_dicts():
            status = self._interpret_state(
                row.get('state_distance', 0),
                row.get('state_velocity', 0),
                row.get('state_acceleration', 0)
            )
            results.append({
                **row,
                'status': status,
                'status_label': self._status_label(status),
            })

        return results

    def summarize_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Generate summary statistics for an entity."""
        df = self.get_state_trajectory(entity_id)

        if df.is_empty():
            return None

        # Get values safely
        distance_col = df.get_column("state_distance")
        velocity_col = df.get_column("state_velocity")

        # Count above thresholds
        warning_count = df.filter(
            pl.col("state_distance") > self.thresholds.distance_warning
        ).height
        critical_count = df.filter(
            pl.col("state_distance") > self.thresholds.distance_critical
        ).height

        return {
            'entity_id': entity_id,
            'n_observations': df.height,
            'I_range': [float(df['I'].min()), float(df['I'].max())],

            # Distance stats
            'distance_mean': float(distance_col.mean()) if distance_col.len() > 0 else 0,
            'distance_max': float(distance_col.max()) if distance_col.len() > 0 else 0,
            'distance_current': float(distance_col[-1]) if distance_col.len() > 0 else 0,

            # Velocity stats
            'velocity_mean': float(velocity_col.mean()) if velocity_col.len() > 0 else 0,
            'velocity_max': float(velocity_col.max()) if velocity_col.len() > 0 else 0,
            'velocity_current': float(velocity_col[-1]) if velocity_col.len() > 0 else 0,

            # Time above thresholds
            'pct_warning': warning_count / df.height * 100 if df.height > 0 else 0,
            'pct_critical': critical_count / df.height * 100 if df.height > 0 else 0,

            # Transitions
            'n_degradation_events': len([
                t for t in self.find_transitions(entity_id)
                if t.get('transition_type') == 'degradation_start'
            ]),

            # Current status
            'current_status': self.get_current_state(entity_id).get('status', 'unknown'),
        }

    def summarize_fleet(self) -> Dict[str, Any]:
        """Generate summary for entire fleet."""
        entities = self.get_entities()
        all_states = self.get_all_current_states()

        # Count by status
        status_counts = {'normal': 0, 'watch': 0, 'warning': 0, 'critical': 0}
        for state in all_states:
            status = state.get('status', 'normal')
            status_counts[status] = status_counts.get(status, 0) + 1

        # Find worst entities
        worst = sorted(all_states, key=lambda x: x.get('state_distance', 0), reverse=True)[:5]

        return {
            'n_entities': len(entities),
            'status_counts': status_counts,
            'pct_healthy': status_counts['normal'] / len(entities) * 100 if entities else 0,
            'worst_entities': [
                {
                    'entity_id': w['entity_id'],
                    'state_distance': w.get('state_distance'),
                    'status': w.get('status'),
                }
                for w in worst
            ],
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_analyzers: Dict[str, StateAnalyzer] = {}


def get_state_analyzer(
    job_id: str = None,
    state_path: Union[str, Path] = None,
    state_df: pl.DataFrame = None,
) -> StateAnalyzer:
    """
    Get or create StateAnalyzer for a job.

    Args:
        job_id: Job ID to look up state.parquet
        state_path: Or provide path directly
        state_df: Or provide DataFrame directly
    """
    if state_df is not None:
        return StateAnalyzer(state_df=state_df)

    if state_path:
        return StateAnalyzer(state_path=state_path)

    if job_id:
        # Check cache
        if job_id in _analyzers:
            return _analyzers[job_id]

        # Look up job output directory
        from prime.services.job_manager import get_job_manager
        manager = get_job_manager()
        job = manager.get_job(job_id)

        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Try PRISM output directory
        prism_output = Path(f"/Users/jasonrudder/prism/data/output/{job_id}")
        state_path = prism_output / "state.parquet"

        if not state_path.exists():
            raise FileNotFoundError(f"state.parquet not found for job {job_id}")

        analyzer = StateAnalyzer(state_path=state_path)
        _analyzers[job_id] = analyzer
        return analyzer

    raise ValueError("Must provide job_id, state_path, or state_df")


__all__ = [
    'StateAnalyzer',
    'StateThresholds',
    'get_state_analyzer',
]
