"""ORTHON Intake - File upload, validation, and transformation (UI handling only)."""

from .upload import (
    load_file,
    detect_format,
    load_matlab_file,
    load_matlab_directory,
    create_observations_parquet,
    SUPPORTED_FORMATS,
)
from .validate import validate, detect_columns, detect_units
from .transformer import (
    prepare_for_prism,
    transform_for_prism,
    IntakeTransformer,
    PrismConfig,
    SignalInfo,
    DISCIPLINES,
    detect_unit,
    strip_unit_suffix,
)

# Re-export manifest models from config (single source of truth)
from ..config.manifest import (
    Manifest,
    PrismManifest,
    EngineManifestEntry,
    ManifestMetadata,
    WindowManifest,
    WindowConfig,
    create_manifest,
    generate_full_manifest,
    ENGINES,
)

__all__ = [
    # Upload
    'load_file',
    'detect_format',
    'load_matlab_file',
    'load_matlab_directory',
    'create_observations_parquet',
    'SUPPORTED_FORMATS',
    # Validate
    'validate',
    'detect_columns',
    'detect_units',
    # Transform
    'prepare_for_prism',
    'transform_for_prism',
    'IntakeTransformer',
    # Config Schema
    'PrismConfig',
    'SignalInfo',
    'DISCIPLINES',
    # Manifest (from config - single source of truth)
    'Manifest',
    'PrismManifest',
    'EngineManifestEntry',
    'ManifestMetadata',
    'WindowManifest',
    'WindowConfig',
    'create_manifest',
    'generate_full_manifest',
    'ENGINES',
    # Utilities
    'detect_unit',
    'strip_unit_suffix',
]
