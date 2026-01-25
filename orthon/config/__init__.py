"""ORTHON configuration module."""

from .recommender import ConfigRecommender, ConfigRecommendation, WindowRecommendation
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
