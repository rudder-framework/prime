"""ORTHON Intake - File upload, validation, and transformation."""

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
from .config_generator import (
    generate_config,
    save_config,
    generate_and_save_config,
    detect_category,
    get_enabled_engines,
    PrismJobConfig,
    SignalConfig,
    UNIT_TO_CATEGORY,
    CORE_ENGINES,
    DOMAIN_ENGINES,
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
    # Config Generator
    'generate_config',
    'save_config',
    'generate_and_save_config',
    'detect_category',
    'get_enabled_engines',
    'PrismJobConfig',
    'SignalConfig',
    'UNIT_TO_CATEGORY',
    'CORE_ENGINES',
    'DOMAIN_ENGINES',
    # Utilities
    'detect_unit',
    'strip_unit_suffix',
]
