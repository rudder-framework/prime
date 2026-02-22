"""
Baseline package for the Rudder Framework.

Two responsibilities:

1. Fleet baseline: eigenstructure computed from early-life pooled data
   across multiple cohorts. The "healthy" reference state.

2. Segment comparison: early vs late geometry delta per cohort.
   How much did the system change from its baseline?
"""

from baseline.reference import compute_fleet_baseline
from baseline.segments import compute_segment_comparison

__all__ = ['compute_fleet_baseline', 'compute_segment_comparison']
