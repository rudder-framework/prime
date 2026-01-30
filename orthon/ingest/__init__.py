"""
ORTHON Universal Data Ingestion

2am grad student provides: folder of data files
ORTHON provides: observations.parquet + manifest.yaml

Output always goes to: /Users/jasonrudder/prism/data/
NO EXCEPTIONS.
"""

from .streaming import ingest_from_manifest, ingest_with_builder
from .manifest_generator import generate_manifest, scan_and_report
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
