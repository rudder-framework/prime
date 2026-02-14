"""Tests for geometry-neutral state classification."""

import pytest
from prime.state import (
    GeometricState,
    StateThresholds,
    WindowMetrics,
    classify_state,
    interpret_state,
    create_interpretation,
    INDUSTRIAL_INTERPRETATION,
    RESEARCH_INTERPRETATION,
    PROCESS_INTERPRETATION,
)


class TestGeometricState:
    """Test GeometricState enum."""

    def test_string_conversion(self):
        """States convert to readable strings."""
        assert str(GeometricState.BASELINE_STABLE) == "BASELINE_STABLE"
        assert str(GeometricState.TRANSITIONING) == "TRANSITIONING"
        assert str(GeometricState.SHIFTED_STABLE) == "SHIFTED_STABLE"
        assert str(GeometricState.INDETERMINATE) == "INDETERMINATE"

    def test_value_access(self):
        """Values match expected strings."""
        assert GeometricState.BASELINE_STABLE.value == "BASELINE_STABLE"


class TestStateThresholds:
    """Test StateThresholds dataclass."""

    def test_default_values(self):
        """Default thresholds are sensible."""
        t = StateThresholds()
        assert t.alignment_stable == 0.95
        assert t.dim_collapse_ratio == 0.70
        assert t.op_shift_sigma == 2.0

    def test_custom_values(self):
        """Custom thresholds can be set."""
        t = StateThresholds(
            alignment_stable=0.90,
            dim_collapse_ratio=0.60,
            op_shift_sigma=3.0,
        )
        assert t.alignment_stable == 0.90
        assert t.dim_collapse_ratio == 0.60
        assert t.op_shift_sigma == 3.0


class TestClassifyState:
    """Test state classification logic."""

    def test_baseline_stable(self):
        """Stable structure at reference operating point."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=5.0,
            eff_dim_ratio=0.95,
            eigenvec_align=0.98,
            mean_op_deviation=0.5,
            max_op_deviation=1.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )
        assert classify_state(metrics) == GeometricState.BASELINE_STABLE

    def test_transitioning_low_alignment(self):
        """Eigenstructure rotating (low alignment)."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=5.0,
            eff_dim_ratio=0.95,
            eigenvec_align=0.85,  # Below threshold
            mean_op_deviation=0.5,
            max_op_deviation=1.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )
        assert classify_state(metrics) == GeometricState.TRANSITIONING

    def test_transitioning_dim_collapse(self):
        """Dimensional collapse."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=2.0,
            eff_dim_ratio=0.5,  # Below threshold
            eigenvec_align=0.98,
            mean_op_deviation=0.5,
            max_op_deviation=1.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )
        assert classify_state(metrics) == GeometricState.TRANSITIONING

    def test_transitioning_both_unstable(self):
        """Both alignment and dimension unstable."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=2.0,
            eff_dim_ratio=0.5,  # Collapsed
            eigenvec_align=0.80,  # Rotating
            mean_op_deviation=0.5,
            max_op_deviation=1.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )
        assert classify_state(metrics) == GeometricState.TRANSITIONING

    def test_shifted_stable(self):
        """Stable at different operating point."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=5.0,
            eff_dim_ratio=0.95,
            eigenvec_align=0.98,
            mean_op_deviation=3.5,  # Above threshold
            max_op_deviation=4.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )
        assert classify_state(metrics) == GeometricState.SHIFTED_STABLE

    def test_custom_thresholds_relaxed_alignment(self):
        """Relaxed alignment threshold changes classification."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=5.0,
            eff_dim_ratio=0.95,
            eigenvec_align=0.92,  # Between 0.90 and 0.95
            mean_op_deviation=0.5,
            max_op_deviation=1.2,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )

        # Default threshold (0.95) → TRANSITIONING
        assert classify_state(metrics) == GeometricState.TRANSITIONING

        # Relaxed threshold (0.90) → BASELINE_STABLE
        relaxed = StateThresholds(alignment_stable=0.90)
        assert classify_state(metrics, relaxed) == GeometricState.BASELINE_STABLE

    def test_custom_thresholds_strict_op_shift(self):
        """Stricter operating point threshold."""
        metrics = WindowMetrics(
            I=100,
            effective_dim=5.0,
            eff_dim_ratio=0.95,
            eigenvec_align=0.98,
            mean_op_deviation=2.5,  # Between 2.0 and 3.0
            max_op_deviation=3.0,
            lambda_1=2.5,
            lambda_1_ratio=1.1,
        )

        # Default threshold (2.0) → SHIFTED_STABLE
        assert classify_state(metrics) == GeometricState.SHIFTED_STABLE

        # Stricter threshold (3.0) → BASELINE_STABLE
        strict = StateThresholds(op_shift_sigma=3.0)
        assert classify_state(metrics, strict) == GeometricState.BASELINE_STABLE


class TestInterpretation:
    """Test domain-specific interpretation."""

    def test_industrial_interpretation(self):
        """Industrial labels map correctly."""
        assert interpret_state(
            GeometricState.BASELINE_STABLE,
            INDUSTRIAL_INTERPRETATION
        ) == "HEALTHY"

        assert interpret_state(
            GeometricState.TRANSITIONING,
            INDUSTRIAL_INTERPRETATION
        ) == "WARNING"

        assert interpret_state(
            GeometricState.SHIFTED_STABLE,
            INDUSTRIAL_INTERPRETATION
        ) == "FAILED"

        assert interpret_state(
            GeometricState.INDETERMINATE,
            INDUSTRIAL_INTERPRETATION
        ) == "UNKNOWN"

    def test_research_interpretation(self):
        """Research labels map correctly."""
        assert interpret_state(
            GeometricState.BASELINE_STABLE,
            RESEARCH_INTERPRETATION
        ) == "INITIAL_STATE"

        assert interpret_state(
            GeometricState.TRANSITIONING,
            RESEARCH_INTERPRETATION
        ) == "TRANSFORMATION"

        assert interpret_state(
            GeometricState.SHIFTED_STABLE,
            RESEARCH_INTERPRETATION
        ) == "ENDPOINT"

        assert interpret_state(
            GeometricState.INDETERMINATE,
            RESEARCH_INTERPRETATION
        ) == "UNCLASSIFIED"

    def test_process_interpretation(self):
        """Process control labels map correctly."""
        assert interpret_state(
            GeometricState.BASELINE_STABLE,
            PROCESS_INTERPRETATION
        ) == "NOMINAL"

        assert interpret_state(
            GeometricState.TRANSITIONING,
            PROCESS_INTERPRETATION
        ) == "TRANSIENT"

        assert interpret_state(
            GeometricState.SHIFTED_STABLE,
            PROCESS_INTERPRETATION
        ) == "OFF_SPEC"

    def test_create_interpretation(self):
        """Custom interpretation can be created."""
        battery_interp = create_interpretation(
            baseline_stable="FRESH",
            transitioning="AGING",
            shifted_stable="DEGRADED",
            indeterminate="UNKNOWN"
        )

        assert interpret_state(
            GeometricState.BASELINE_STABLE, battery_interp
        ) == "FRESH"

        assert interpret_state(
            GeometricState.SHIFTED_STABLE, battery_interp
        ) == "DEGRADED"

    def test_missing_interpretation_falls_back(self):
        """Missing mapping falls back to state string."""
        partial_interp = {
            GeometricState.BASELINE_STABLE: "OK",
            # TRANSITIONING not defined
        }

        assert interpret_state(
            GeometricState.BASELINE_STABLE, partial_interp
        ) == "OK"

        # Falls back to enum string
        assert interpret_state(
            GeometricState.TRANSITIONING, partial_interp
        ) == "TRANSITIONING"


class TestWindowMetrics:
    """Test WindowMetrics dataclass."""

    def test_creation(self):
        """Metrics can be created with all fields."""
        m = WindowMetrics(
            I=100,
            effective_dim=3.5,
            eff_dim_ratio=0.85,
            eigenvec_align=0.92,
            mean_op_deviation=1.5,
            max_op_deviation=2.3,
            lambda_1=2.1,
            lambda_1_ratio=1.3,
        )
        assert m.I == 100
        assert m.effective_dim == 3.5
        assert m.eff_dim_ratio == 0.85
