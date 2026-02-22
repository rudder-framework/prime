"""
Topology package for the Rudder Framework.

Topological Data Analysis via persistent homology.
Computes Betti numbers and persistence diagrams from point clouds.

Betti-0: connected components (how fragmented is the state space?)
Betti-1: loops/cycles (recurrent patterns?)
Betti-2: voids (enclosed regions?)

Persistence = lifetime of topological features. Long-lived features
are genuine structure; short-lived are noise.
"""

from topology.homology import compute_persistence, betti_numbers_at_threshold

__all__ = ['compute_persistence', 'betti_numbers_at_threshold']
