"""
Rudder Framework - Signal Classification and Manifest Generation

Architecture (v2.5):
- Typology-guided, scale-invariant, multi-scale representation
- Per-engine window specification
"""

__version__ = "2.5.0"

# Lazy imports to avoid circular dependencies
# Use: from framework.core import Pipeline, DataReader, PrismClient
# Use: from framework.manifest import build_manifest, save_manifest
# Use: from framework.ingest import transform_to_prism_format
# Use: from framework.typology import apply_discrete_sparse_classification
# Use: from framework.cohorts import discover_cohorts

__all__ = [
    '__version__',
]
