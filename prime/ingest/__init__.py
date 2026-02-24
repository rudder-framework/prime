"""
Universal Data Ingestion

Schema v2.6:
- REQUIRED: signal_id, signal_0, value
- OPTIONAL: cohort (grouping key, replaces unit_id)
"""

from .normalize import normalize_observations, _METADATA_COLUMNS, _GPS_COLUMNS
from .transform import (
    validate_manifold_schema,
    transform_wide_to_long,
    ensure_signal_0_sorted,
    transform_to_manifold_format,
    transform_femto,
    transform_skab,
    transform_fama_french,
    transform_cmapss,
)
from .validate_observations import validate_and_save, validate_observations, ValidationStatus
from .paths import (
    OUTPUT_DIR,
    OBSERVATIONS_PATH,
    MANIFEST_PATH,
    get_observations_path,
    get_manifest_path,
    ensure_output_dir,
    check_output_empty,
    list_output_contents,
)
from .typology_raw import compute_typology_raw
from .upload import load_file, preview_file, get_file_info, create_observations_parquet
from .regime_normalize import regime_normalize, RegimeNormalizationResult

# Alias for transform_to_manifold_format
transform_to_canonical = transform_to_manifold_format

__all__ = [
    # Normalize functions
    "normalize_observations",
    "_METADATA_COLUMNS",
    "_GPS_COLUMNS",
    # Transform functions
    "validate_manifold_schema",
    "transform_wide_to_long",
    "ensure_signal_0_sorted",
    "transform_to_manifold_format",
    "transform_to_canonical",
    "transform_femto",
    "transform_skab",
    "transform_fama_french",
    "transform_cmapss",
    # Validation
    "validate_observations",
    "validate_and_save",
    "ValidationStatus",
    # Paths (fixed)
    "OUTPUT_DIR",
    "OBSERVATIONS_PATH",
    "MANIFEST_PATH",
    "get_observations_path",
    "get_manifest_path",
    "ensure_output_dir",
    "check_output_empty",
    "list_output_contents",
    # Typology
    "compute_typology_raw",
    # Upload/load
    "load_file",
    "preview_file",
    "get_file_info",
    "create_observations_parquet",
    # Regime normalization
    "regime_normalize",
    "RegimeNormalizationResult",
]
