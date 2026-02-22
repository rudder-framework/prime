"""
Fleet package for the Rudder Framework.

Cross-cohort analysis: treats cohort centroids as signals
and applies the same eigendecomp/pairwise/velocity machinery
at the fleet scale.

Fleet eigendecomp: how are cohorts distributed in feature space?
Fleet pairwise: which cohorts are similar/diverging?
Fleet velocity: how is the fleet evolving?
"""

from fleet.analysis import (
    compute_fleet_eigendecomp,
    compute_fleet_pairwise,
    compute_fleet_velocity,
)

__all__ = [
    'compute_fleet_eigendecomp',
    'compute_fleet_pairwise',
    'compute_fleet_velocity',
]
