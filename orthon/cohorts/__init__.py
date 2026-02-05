"""
Cohort discovery and baseline detection.
"""

from .discovery import (
    CohortDiscovery,
    CohortResult,
    SignalClassification,
    process_observations,
)
from .detection import (
    detect_cohorts,
    detect_constants,
    CohortType,
    should_run_cohort_discovery,
)
from .baseline import (
    discover_stable_baseline,
    get_baseline,
    compute_deviation,
    BaselineMode,
    BaselineResult,
)

# Aliases for backward compatibility
discover_cohorts = process_observations
discover_baseline = discover_stable_baseline

__all__ = [
    # Discovery
    'CohortDiscovery',
    'CohortResult',
    'SignalClassification',
    'process_observations',
    'discover_cohorts',
    # Detection
    'detect_cohorts',
    'detect_constants',
    'CohortType',
    'should_run_cohort_discovery',
    # Baseline
    'discover_stable_baseline',
    'discover_baseline',
    'get_baseline',
    'compute_deviation',
    'BaselineMode',
    'BaselineResult',
]
