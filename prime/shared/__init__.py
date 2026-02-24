"""Prime Shared — Schema definitions shared with Manifold."""

from .config_schema import (
    DISCIPLINES,
    DisciplineType,
    DOMAIN_TO_DISCIPLINE,
    DomainType,
)
from .physics_constants import (
    PhysicsConstants,
    ENERGY_FORMULAS,
    VALID_UNIT_CATEGORIES,
    can_compute_real_energy,
    get_unit_category,
)

__all__ = [
    # Disciplines
    'DISCIPLINES',
    'DisciplineType',
    'DOMAIN_TO_DISCIPLINE',
    'DomainType',
    # Physics Constants
    'PhysicsConstants',
    'ENERGY_FORMULAS',
    'VALID_UNIT_CATEGORIES',
    'can_compute_real_energy',
    'get_unit_category',
]
