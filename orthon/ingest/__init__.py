"""
ORTHON Universal Data Ingestion

Schema v2.0.0:
- REQUIRED: signal_id, I, value
- OPTIONAL: unit_id (just a label, blank is fine)

Output always goes to: /Users/jasonrudder/prism/data/
"""

from .streaming import ingest_from_manifest, ingest_with_builder
from .manifest_generator import generate_manifest, scan_and_report
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

__all__ = [
    # Ingest functions
    "ingest_from_manifest",
    "ingest_with_builder",
    # Manifest generation
    "generate_manifest",
    "scan_and_report",
    # Transform functions
    "validate_prism_schema",
    "transform_wide_to_long",
    "fix_sparse_index",
    "transform_to_prism_format",
    "transform_femto",
    "transform_skab",
    "transform_fama_french",
    "transform_cmapss",
    # Validation
    "validate_observations",
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
]
