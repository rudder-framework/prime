"""
Geometry package for the Rudder Framework.

Two responsibilities:

1. Signal geometry: Where is each signal relative to the eigenstructure?
   - Distance to centroid
   - Coherence to PC1 (alignment with dominant direction)
   - Contribution (projection onto centroid direction)
   - Residual (orthogonal component)

2. Eigenvalue dynamics: How is the geometry changing over time?
   - Velocity/acceleration/jerk of effective_dim and eigenvalues
   - Curvature of the eigenvalue trajectory
   - Collapse detection (sustained dimensional loss)

This is the package that connects eigendecomp (the shape) to prediction
(the trajectory). Failure = dimensional collapse over time.
"""

from geometry.signal import (
    compute_signal_geometry,
    compute_signal_geometry_batch,
)
from geometry.dynamics import (
    compute_derivatives,
    compute_eigenvalue_dynamics,
)
from geometry.collapse import (
    detect_collapse,
)

__all__ = [
    'compute_signal_geometry',
    'compute_signal_geometry_batch',
    'compute_derivatives',
    'compute_eigenvalue_dynamics',
    'detect_collapse',
]
