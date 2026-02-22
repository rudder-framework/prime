"""
Dynamics package for the Rudder Framework.

Computes Finite-Time Lyapunov Exponents (FTLE) per signal.
FTLE measures trajectory sensitivity over finite windows — the correct
framing for non-stationary industrial data.

FTLE > 0: trajectories diverge (instability)
FTLE ≈ 0: trajectories parallel (quasi-periodic)
FTLE < 0: trajectories converge (stability)

Rolling FTLE tracks stability evolution over time.
"""

from dynamics.ftle import compute_ftle, compute_ftle_rolling

__all__ = ['compute_ftle', 'compute_ftle_rolling']
