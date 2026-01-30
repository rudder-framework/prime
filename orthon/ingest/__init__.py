"""
ORTHON Universal Data Ingestion

2am grad student provides: folder of data files
ORTHON provides: observations.parquet + manifest.yaml
"""

from .streaming import ingest_from_manifest
from .manifest_generator import generate_manifest

__all__ = ["ingest_from_manifest", "generate_manifest"]
