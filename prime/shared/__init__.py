"""Prime Shared — Schema definitions shared with PRISM."""

from .config_schema import (
    PrismConfig,
    SignalInfo,
    WindowConfig as WindowConfigModel,
    BaselineConfig,
    RegimeConfig,
    StateConfig,
    DISCIPLINES,
    DisciplineType,
    DOMAIN_TO_DISCIPLINE,
    DomainType,
)
from .window_config import (
    WindowConfig,
    auto_detect_window,
    validate_window,
    get_recommendation,
    format_errors_for_ui,
    format_config_summary,
    DOMAIN_DEFAULTS,
    COMPUTE_LIMITS,
)
from .engine_registry import (
    Granularity,
    Pillar,
    EngineSpec,
    UNIT_TO_CATEGORY,
    ENGINE_SPECS,
    get_engines_for_categories,
    get_universal_engines,
    get_domain_engines,
    get_engines_by_pillar,
    get_category_for_unit,
)
from .physics_constants import (
    PhysicsConstants,
    ENERGY_FORMULAS,
    VALID_UNIT_CATEGORIES,
    can_compute_real_energy,
    get_unit_category,
)

__all__ = [
    # Config Schema
    'PrismConfig',
    'SignalInfo',
    'WindowConfigModel',
    'BaselineConfig',
    'RegimeConfig',
    'StateConfig',
    # Disciplines
    'DISCIPLINES',
    'DisciplineType',
    'DOMAIN_TO_DISCIPLINE',
    'DomainType',
    # Window Config (auto-detection)
    'WindowConfig',
    'auto_detect_window',
    'validate_window',
    'get_recommendation',
    'format_errors_for_ui',
    'format_config_summary',
    'DOMAIN_DEFAULTS',
    'COMPUTE_LIMITS',
    # Engine Registry
    'Granularity',
    'Pillar',
    'EngineSpec',
    'UNIT_TO_CATEGORY',
    'ENGINE_SPECS',
    'get_engines_for_categories',
    'get_universal_engines',
    'get_domain_engines',
    'get_engines_by_pillar',
    'get_category_for_unit',
    # Physics Constants
    'PhysicsConstants',
    'ENERGY_FORMULAS',
    'VALID_UNIT_CATEGORIES',
    'can_compute_real_energy',
    'get_unit_category',
]
