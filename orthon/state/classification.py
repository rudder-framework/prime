"""
State classification based on eigenstructure geometry.

Labels are geometry-neutral. Interpretation (healthy/failed,
initial/endpoint) is domain-specific and left to the user.

States:
    BASELINE_STABLE  - Structure matches reference, stable eigenstructure
    TRANSITIONING    - Eigenstructure rotating or dimensions collapsing
    SHIFTED_STABLE   - Stable eigenstructure at different operating point
    INDETERMINATE    - Cannot reliably classify

Philosophy:
    ORTHON measures geometry. The eigenstructure doesn't know if dimensional
    collapse is "failure" or "transformation" - that's interpretation.

    A bearing losing coherence is failure.
    A plastic film losing coherence is the experiment.
    Same math. Different meaning.

    "geometry leads - orthon"
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict


class GeometricState(str, Enum):
    """Geometry-neutral state classification."""

    BASELINE_STABLE = "BASELINE_STABLE"
    TRANSITIONING = "TRANSITIONING"
    SHIFTED_STABLE = "SHIFTED_STABLE"
    INDETERMINATE = "INDETERMINATE"

    def __str__(self) -> str:
        return self.value


@dataclass
class StateThresholds:
    """
    Thresholds for state classification.

    Users can adjust based on domain requirements.

    Attributes:
        alignment_stable: Eigenvector alignment threshold for stability
        dim_collapse_ratio: eff_dim / baseline_eff_dim below which = collapse
        op_shift_sigma: Operating point deviation threshold in baseline σ
    """
    alignment_stable: float = 0.95
    dim_collapse_ratio: float = 0.70
    op_shift_sigma: float = 2.0


@dataclass
class WindowMetrics:
    """
    Metrics for a single analysis window.

    These are computed by the dimensional analysis and driving vector
    analysis scripts, then used for state classification.
    """
    I: int
    effective_dim: float
    eff_dim_ratio: float
    eigenvec_align: float
    mean_op_deviation: float
    max_op_deviation: float
    lambda_1: float
    lambda_1_ratio: float


def classify_state(
    metrics: WindowMetrics,
    thresholds: Optional[StateThresholds] = None
) -> GeometricState:
    """
    Classify window state based on geometric metrics.

    Decision logic:
        1. Is eigenstructure stable? (alignment > threshold)
        2. Is dimensionality collapsing? (eff_dim_ratio < threshold)
        3. Is operating point shifted? (deviation > threshold)

    Classification matrix:
        stable  collapse  shifted  →  state
        ─────────────────────────────────────
        yes     no        no       →  BASELINE_STABLE
        no      -         -        →  TRANSITIONING
        -       yes       -        →  TRANSITIONING
        yes     no        yes      →  SHIFTED_STABLE
        other                      →  INDETERMINATE

    Args:
        metrics: Window metrics from eigenanalysis
        thresholds: Classification thresholds (uses defaults if None)

    Returns:
        GeometricState classification
    """
    if thresholds is None:
        thresholds = StateThresholds()

    # Assess stability
    structure_stable = metrics.eigenvec_align > thresholds.alignment_stable
    dim_collapsing = metrics.eff_dim_ratio < thresholds.dim_collapse_ratio
    op_shifted = metrics.mean_op_deviation > thresholds.op_shift_sigma

    # Classification logic
    if structure_stable and not dim_collapsing and not op_shifted:
        return GeometricState.BASELINE_STABLE

    elif not structure_stable or dim_collapsing:
        return GeometricState.TRANSITIONING

    elif structure_stable and op_shifted:
        return GeometricState.SHIFTED_STABLE

    else:
        return GeometricState.INDETERMINATE


# =============================================================================
# Domain-specific interpretation helpers
# =============================================================================
# Users can define their own mappings. These are provided as examples.

INDUSTRIAL_INTERPRETATION: Dict[GeometricState, str] = {
    GeometricState.BASELINE_STABLE: "HEALTHY",
    GeometricState.TRANSITIONING: "WARNING",
    GeometricState.SHIFTED_STABLE: "FAILED",
    GeometricState.INDETERMINATE: "UNKNOWN",
}

RESEARCH_INTERPRETATION: Dict[GeometricState, str] = {
    GeometricState.BASELINE_STABLE: "INITIAL_STATE",
    GeometricState.TRANSITIONING: "TRANSFORMATION",
    GeometricState.SHIFTED_STABLE: "ENDPOINT",
    GeometricState.INDETERMINATE: "UNCLASSIFIED",
}

PROCESS_INTERPRETATION: Dict[GeometricState, str] = {
    GeometricState.BASELINE_STABLE: "NOMINAL",
    GeometricState.TRANSITIONING: "TRANSIENT",
    GeometricState.SHIFTED_STABLE: "OFF_SPEC",
    GeometricState.INDETERMINATE: "UNDEFINED",
}


def interpret_state(
    state: GeometricState,
    interpretation: Dict[GeometricState, str]
) -> str:
    """
    Apply domain-specific interpretation to geometric state.

    Args:
        state: Neutral geometric state
        interpretation: Mapping from GeometricState to domain label

    Returns:
        Domain-specific label string

    Example:
        >>> state = GeometricState.SHIFTED_STABLE
        >>> interpret_state(state, INDUSTRIAL_INTERPRETATION)
        'FAILED'
        >>> interpret_state(state, RESEARCH_INTERPRETATION)
        'ENDPOINT'
    """
    return interpretation.get(state, str(state))


def create_interpretation(
    baseline_stable: str = "BASELINE_STABLE",
    transitioning: str = "TRANSITIONING",
    shifted_stable: str = "SHIFTED_STABLE",
    indeterminate: str = "INDETERMINATE",
) -> Dict[GeometricState, str]:
    """
    Create a custom interpretation mapping.

    Args:
        baseline_stable: Label for BASELINE_STABLE state
        transitioning: Label for TRANSITIONING state
        shifted_stable: Label for SHIFTED_STABLE state
        indeterminate: Label for INDETERMINATE state

    Returns:
        Dict mapping GeometricState to custom labels

    Example:
        >>> battery_interp = create_interpretation(
        ...     baseline_stable="FRESH",
        ...     transitioning="AGING",
        ...     shifted_stable="DEGRADED",
        ...     indeterminate="UNKNOWN"
        ... )
    """
    return {
        GeometricState.BASELINE_STABLE: baseline_stable,
        GeometricState.TRANSITIONING: transitioning,
        GeometricState.SHIFTED_STABLE: shifted_stable,
        GeometricState.INDETERMINATE: indeterminate,
    }
