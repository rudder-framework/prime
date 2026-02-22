"""
Velocity package for the Rudder Framework.

Computes the velocity vector of the system state at each index.
Works on observation-level data (pivoted wide: rows=time, cols=signals).

Outputs: speed, direction, acceleration, curvature, dominant motion signal,
motion dimensionality.
"""

from velocity.field import compute_velocity_field

__all__ = ['compute_velocity_field']
