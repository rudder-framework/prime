"""
Prime Explorer Models
======================

Data structures for manifold visualization.
Zero calculations — just containers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EntityState:
    """State of one entity at current view."""
    entity_id: str

    # Position (3D projection from PCA)
    position: np.ndarray  # [x, y, z]

    # Motion (from dynamics)
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    hd_slope: float  # THE metric

    # Forces (from physics)
    force: np.ndarray  # [fx, fy, fz]
    kinetic_energy: float
    potential_energy: float

    # Classification
    regime: int  # 0=healthy, 1=degraded, 2=critical
    regime_name: str

    # Visual
    color: Tuple[float, float, float]  # RGB
    size: float

    # Trail
    trajectory: np.ndarray  # (N, 3) history


@dataclass
class ManifoldState:
    """Complete state for one view."""
    window_id: int

    # All entities
    entities: Dict[str, EntityState]

    # Structure (from geometry)
    regime_centers: np.ndarray  # (n_regimes, 3)
    regime_boundaries: List[np.ndarray]

    # Attractors (from physics)
    healthy_attractor: np.ndarray
    failure_attractor: np.ndarray

    # Stats
    n_healthy: int
    n_degraded: int
    n_critical: int
    mean_hd_slope: float


@dataclass
class ExplorerConfig:
    """Display settings."""
    # Thresholds
    hd_slope_healthy: float = -0.01
    hd_slope_warning: float = -0.05
    hd_slope_critical: float = -0.10

    # Display toggles
    show_velocities: bool = True
    show_forces: bool = True
    show_trajectories: bool = True
    show_labels: bool = True
    trajectory_length: int = 10
