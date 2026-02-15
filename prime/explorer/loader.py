"""
Prime Manifold Loader
======================

Load Manifold parquet outputs for visualization.
Zero calculations — just read and structure.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json

from .models import EntityState, ManifoldState, ExplorerConfig


class ManifoldLoader:
    """Load Manifold outputs for visualization."""

    def __init__(self, data_dir: Path, config: ExplorerConfig = None):
        self.data_dir = Path(data_dir)
        self.config = config or ExplorerConfig()

        # Load all parquets
        self.geometry = self._load_parquet('geometry.parquet')
        self.dynamics = self._load_parquet('dynamics.parquet')
        self.physics = self._load_parquet('physics.parquet')

        # Validate
        self._validate()

        # Load projection matrix
        self.pca_matrix = self._load_pca_matrix()

    def _load_parquet(self, name: str) -> pl.DataFrame:
        """Load a parquet file if it exists."""
        path = self.data_dir / name
        if path.exists():
            return pl.read_parquet(path)
        return pl.DataFrame()

    def _validate(self):
        """Validate Manifold output structure."""
        if self.dynamics.is_empty():
            raise FileNotFoundError(
                f"dynamics.parquet not found in {self.data_dir}. "
                "This file is required for manifold visualization."
            )

        # dynamics must have hd_slope
        if 'hd_slope' not in self.dynamics.columns:
            raise ValueError(
                "dynamics.parquet missing 'hd_slope' column.\n"
                "This is THE key metric. Manifold must compute it."
            )

        # Validate: dynamics should have ONE row per entity (if geometry exists)
        if not self.geometry.is_empty():
            n_entities = self.geometry['entity_id'].n_unique()
            n_dynamics = len(self.dynamics)
            if n_dynamics != n_entities:
                print(
                    f"Warning: dynamics.parquet has {n_dynamics} rows, "
                    f"geometry has {n_entities} entities. "
                    f"Expected one dynamics row per entity."
                )

    def _load_pca_matrix(self) -> np.ndarray:
        """Get PCA components for 3D projection."""
        if not self.geometry.is_empty() and 'pca_components' in self.geometry.columns:
            blob = self.geometry['pca_components'][0]
            if isinstance(blob, str):
                components = np.array(json.loads(blob))
            elif blob is not None:
                components = np.array(blob)
            else:
                return np.eye(3)
            return components[:3] if len(components) >= 3 else np.eye(3)
        return np.eye(3)

    def get_entities(self) -> List[str]:
        """Get all entity IDs."""
        return self.dynamics['entity_id'].unique().sort().to_list()

    def load_state(self) -> ManifoldState:
        """Load complete manifold state."""

        entities = {}

        for row in self.dynamics.iter_rows(named=True):
            entity_id = row['entity_id']

            # Get physics row if available
            phys_row = {}
            if not self.physics.is_empty():
                phys = self.physics.filter(
                    pl.col('entity_id') == entity_id
                ).to_dicts()
                phys_row = phys[0] if phys else {}

            # Build entity state
            entities[entity_id] = self._build_entity(row, phys_row)

        # Extract structure from geometry
        regime_centers = self._get_regime_centers()
        healthy_attractor, failure_attractor = self._get_attractors()

        # Count by regime
        n_healthy = sum(1 for e in entities.values() if e.regime == 0)
        n_degraded = sum(1 for e in entities.values() if e.regime == 1)
        n_critical = sum(1 for e in entities.values() if e.regime == 2)

        return ManifoldState(
            window_id=0,
            entities=entities,
            regime_centers=regime_centers,
            regime_boundaries=[],
            healthy_attractor=healthy_attractor,
            failure_attractor=failure_attractor,
            n_healthy=n_healthy,
            n_degraded=n_degraded,
            n_critical=n_critical,
            mean_hd_slope=np.mean([e.hd_slope for e in entities.values()]),
        )

    def _build_entity(self, dyn: dict, phys: dict) -> EntityState:
        """Build EntityState from row data."""

        hd_slope = dyn['hd_slope']

        # Position (from dynamics or compute from projection)
        position = np.array([
            dyn.get('position_pc1', 0) or 0,
            dyn.get('position_pc2', 0) or 0,
            dyn.get('position_pc3', 0) or 0,
        ])

        # Velocity
        velocity = np.array([
            dyn.get('velocity_pc1', dyn.get('hd_velocity_mean', 0)) or 0,
            dyn.get('velocity_pc2', 0) or 0,
            dyn.get('velocity_pc3', 0) or 0,
        ])

        # Acceleration
        acceleration = np.array([
            dyn.get('acceleration_pc1', dyn.get('hd_acceleration_mean', 0)) or 0,
            dyn.get('acceleration_pc2', 0) or 0,
            dyn.get('acceleration_pc3', 0) or 0,
        ])

        # Force
        force = np.array([
            phys.get('force_pc1', 0) or 0,
            phys.get('force_pc2', 0) or 0,
            phys.get('force_pc3', 0) or 0,
        ])

        # Regime
        if hd_slope > self.config.hd_slope_healthy:
            regime, regime_name = 0, 'healthy'
        elif hd_slope > self.config.hd_slope_critical:
            regime, regime_name = 1, 'degraded'
        else:
            regime, regime_name = 2, 'critical'

        # Color (green → yellow → red)
        color = self._hd_slope_to_color(hd_slope)

        # Size (larger if more critical)
        size = 0.2 + abs(hd_slope) * 3

        return EntityState(
            entity_id=dyn['entity_id'],
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            hd_slope=hd_slope,
            force=force,
            kinetic_energy=phys.get('hamiltonian_T', 0) or 0,
            potential_energy=phys.get('hamiltonian_V', 0) or 0,
            regime=regime,
            regime_name=regime_name,
            color=color,
            size=size,
            trajectory=np.array([position]),  # Single point for now
        )

    def _hd_slope_to_color(self, hd_slope: float) -> Tuple[float, float, float]:
        """Map hd_slope to RGB color."""
        cfg = self.config

        if hd_slope > cfg.hd_slope_healthy:
            return (0.2, 0.8, 0.2)  # Green
        elif hd_slope > cfg.hd_slope_warning:
            # Green to yellow
            t = (hd_slope - cfg.hd_slope_warning) / (cfg.hd_slope_healthy - cfg.hd_slope_warning)
            return (0.2 + 0.6 * (1 - t), 0.8, 0.2 * t)
        elif hd_slope > cfg.hd_slope_critical:
            # Yellow to red
            t = (hd_slope - cfg.hd_slope_critical) / (cfg.hd_slope_warning - cfg.hd_slope_critical)
            return (0.9, 0.8 * t, 0.1)
        else:
            return (0.9, 0.1, 0.1)  # Red

    def _get_regime_centers(self) -> np.ndarray:
        """Extract regime cluster centers."""
        if not self.geometry.is_empty() and 'cluster_centers' in self.geometry.columns:
            blob = self.geometry['cluster_centers'][0]
            if blob is not None:
                return np.array(json.loads(blob) if isinstance(blob, str) else blob)
        return np.array([[0, 0, 0], [5, 5, 5]])

    def _get_attractors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get healthy and failure attractors."""
        # Healthy = center of healthy cluster
        # Failure = where critical entities converge
        return np.array([0, 0, 0]), np.array([10, 10, 10])
