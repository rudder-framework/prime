"""ORTHON configuration module."""

from .recommender import ConfigRecommender, ConfigRecommendation, WindowRecommendation
from .manifest import (
    # Engine list
    ENGINES,
    DEFAULT_PRISM_CONFIG,
    # Pydantic models
    Manifest,
    PrismManifest,
    PRISMConfig,
    DataConfig,
    DatasetConfig,
    WindowConfig,
    WindowManifest,
    RAMConfig,
    ComputeConfig,
    EngineManifestEntry,
    ManifestMetadata,
    # Factory functions
    create_manifest,
    generate_full_manifest,
    # Validation
    validate_manifest,
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
    # Full Compute Manifest
    "ENGINES",
    "DEFAULT_PRISM_CONFIG",
    "Manifest",
    "PrismManifest",
    "PRISMConfig",
    "DataConfig",
    "DatasetConfig",
    "WindowConfig",
    "WindowManifest",
    "RAMConfig",
    "ComputeConfig",
    "EngineManifestEntry",
    "ManifestMetadata",
    "create_manifest",
    "generate_full_manifest",
    "validate_manifest",
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
