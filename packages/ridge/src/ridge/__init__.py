"""
Ridge package for the Rudder Framework.

Convergence of FTLE field + velocity field.
Urgency = v · ∇FTLE = rate of approach to regime boundary.

The WARNING quadrant (low FTLE, positive urgency) is the early warning signal.
"""

from ridge.proximity import compute_ridge_proximity

__all__ = ['compute_ridge_proximity']
