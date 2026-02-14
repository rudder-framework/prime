"""
State analysis and classification.

Provides geometry-neutral state classification based on eigenstructure
metrics. Labels describe what the math shows without implying domain-specific
meaning (healthy/failed, initial/endpoint, etc.).

Example:
    >>> from prime.state import classify_state, WindowMetrics, GeometricState
    >>> from prime.state import interpret_state, INDUSTRIAL_INTERPRETATION
    >>>
    >>> metrics = WindowMetrics(
    ...     I=100,
    ...     effective_dim=3.5,
    ...     eff_dim_ratio=0.85,
    ...     eigenvec_align=0.92,
    ...     mean_op_deviation=2.5,
    ...     max_op_deviation=3.1,
    ...     lambda_1=1.8,
    ...     lambda_1_ratio=1.2,
    ... )
    >>> state = classify_state(metrics)
    >>> print(state)  # TRANSITIONING (neutral)
    >>> print(interpret_state(state, INDUSTRIAL_INTERPRETATION))  # WARNING
"""

from .classification import (
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

__all__ = [
    # Core types
    'GeometricState',
    'StateThresholds',
    'WindowMetrics',
    # Classification
    'classify_state',
    # Interpretation
    'interpret_state',
    'create_interpretation',
    # Built-in interpretations
    'INDUSTRIAL_INTERPRETATION',
    'RESEARCH_INTERPRETATION',
    'PROCESS_INTERPRETATION',
]
