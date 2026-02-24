"""Prime configuration module."""

from .recommender import ConfigRecommender, ConfigRecommendation, WindowRecommendation
from .typology_config import (
    TYPOLOGY_CONFIG,
    get_threshold,
    validate_config,
)
from .discrete_sparse_config import (
    DISCRETE_SPARSE_CONFIG,
    DISCRETE_SPARSE_SPECTRAL,
    DISCRETE_SPARSE_ENGINES,
)
from .domains import (
    DOMAINS,
    EQUATION_INFO,
    get_required_inputs,
    get_equations_for_domain,
    validate_inputs,
    generate_config,
)

__all__ = [
    # Typology Config
    "TYPOLOGY_CONFIG",
    "get_threshold",
    "validate_config",
    # Discrete/Sparse Config (PR5)
    "DISCRETE_SPARSE_CONFIG",
    "DISCRETE_SPARSE_SPECTRAL",
    "DISCRETE_SPARSE_ENGINES",
    # Recommender
    "ConfigRecommender",
    "ConfigRecommendation",
    "WindowRecommendation",
    # Domains
    "DOMAINS",
    "EQUATION_INFO",
    "get_required_inputs",
    "get_equations_for_domain",
    "validate_inputs",
    "generate_config",
]
