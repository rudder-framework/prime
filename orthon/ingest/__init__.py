"""
ORTHON Universal Data Ingestion

Schema v2.5:
- REQUIRED: signal_id, I, value
- OPTIONAL: cohort (grouping key, replaces unit_id)

Output always goes to: /Users/jasonrudder/prism/data/
"""

from .streaming import ingest_from_manifest, ingest_with_builder
from .transform import (
    validate_prism_schema,
    transform_wide_to_long,
    fix_sparse_index,
    transform_to_prism_format,
    transform_femto,
    transform_skab,
    transform_fama_french,
    transform_cmapss,
)
from .validate import validate_observations
from .validate_observations import validate_and_save, ValidationStatus
from .data_confirmation import (
    confirm_data,
    confirm_for_api,
    confirm_for_ai,
    ConfirmationResult,
)
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

# Alias for transform_to_prism_format
transform_to_canonical = transform_to_prism_format

__all__ = [
    # Ingest functions
    "ingest_from_manifest",
    "ingest_with_builder",
    # Transform functions
    "validate_prism_schema",
    "transform_wide_to_long",
    "fix_sparse_index",
    "transform_to_prism_format",
    "transform_to_canonical",
    "transform_femto",
    "transform_skab",
    "transform_fama_french",
    "transform_cmapss",
    # Validation
    "validate_observations",
    "validate_and_save",
    "ValidationStatus",
    # Data Confirmation (schema-based)
    "confirm_data",
    "confirm_for_api",
    "confirm_for_ai",
    "ConfirmationResult",
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
]
