"""
ORTHON - Signal Classification and Manifest Generation

ORTHON classifies. PRISM computes.

Architecture (v2.5):
- Typology-guided, scale-invariant, multi-scale representation
- Per-engine window specification
"""

__version__ = "2.5.0"

# Lazy imports to avoid circular dependencies
# Use: from orthon.core import Pipeline, DataReader, PrismClient
# Use: from orthon.manifest import build_manifest, save_manifest
# Use: from orthon.ingest import transform_to_prism_format
# Use: from orthon.typology import apply_discrete_sparse_classification
# Use: from orthon.cohorts import discover_cohorts

__all__ = [
    '__version__',
]
