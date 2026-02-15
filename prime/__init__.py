"""
Prime - Signal Classification and Manifest Generation

Architecture (v2.5):
- Typology-guided, scale-invariant, multi-scale representation
- Per-engine window specification
"""

__version__ = "2.5.0"

# Lazy imports to avoid circular dependencies
# Use: from prime.core import Pipeline, DataReader, ManifoldClient
# Use: from prime.manifest import build_manifest, save_manifest
# Use: from prime.ingest import transform_to_manifold_format
# Use: from prime.typology import apply_discrete_sparse_classification
# Use: from prime.cohorts import discover_cohorts

__all__ = [
    '__version__',
]
