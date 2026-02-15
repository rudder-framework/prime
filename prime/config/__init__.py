"""Prime configuration module."""

from .recommender import ConfigRecommender, ConfigRecommendation, WindowRecommendation
from .typology_config import (
    TYPOLOGY_CONFIG,
    get_threshold,
    get_engine_adjustments,
    get_viz_adjustments,
    validate_config,
)
from .discrete_sparse_config import (
    DISCRETE_SPARSE_CONFIG,
    DISCRETE_SPARSE_SPECTRAL,
    DISCRETE_SPARSE_ENGINES,
    get_discrete_threshold,
)
from .domains import (
    DOMAINS,
    EQUATION_INFO,
    INPUT_DEFINITIONS,
    Capability,
    CAPABILITY_REQUIREMENTS,
    get_required_inputs,
    get_equations_for_domain,
    validate_inputs,
    generate_config,
    # Entry point exports
    turbofan,
    bearings,
    chemical,
    hydraulic,
)

__all__ = [
    # Typology Config
    "TYPOLOGY_CONFIG",
    "get_threshold",
    "get_engine_adjustments",
    "get_viz_adjustments",
    "validate_config",
    # Discrete/Sparse Config (PR5)
    "DISCRETE_SPARSE_CONFIG",
    "DISCRETE_SPARSE_SPECTRAL",
    "DISCRETE_SPARSE_ENGINES",
    "get_discrete_threshold",
    # Recommender
    "ConfigRecommender",
    "ConfigRecommendation",
    "WindowRecommendation",
    # Domains
    "DOMAINS",
    "EQUATION_INFO",
    "INPUT_DEFINITIONS",
    "Capability",
    "CAPABILITY_REQUIREMENTS",
    "get_required_inputs",
    "get_equations_for_domain",
    "validate_inputs",
    "generate_config",
    "turbofan",
    "bearings",
    "chemical",
    "hydraulic",
]
