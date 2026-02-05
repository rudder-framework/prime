"""
ORTHON Entry Points - Ordered Pipeline Stages
==============================================

Thin orchestrators that call engines/modules for computation.
Entry points do NOT contain compute logic - only orchestration.

Pipeline Order:
    stage_01_validate   → Validation (remove constants, duplicates)
    stage_02_typology   → Compute raw typology measures (27 metrics)
    stage_03_classify   → Apply classification (discrete/sparse → continuous)
    stage_04_manifest   → Generate manifest for PRISM
    stage_05_diagnostic → Run diagnostic assessment (uses engines)

Usage:
    python -m orthon.entry_points.stage_01_validate observations.parquet -o validated.parquet
    python -m orthon.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
    python -m orthon.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
    python -m orthon.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
    python -m orthon.entry_points.stage_05_diagnostic observations.parquet -o report.txt

PRISM computes numbers. ORTHON classifies.
"""

# Import run functions for convenience
from .stage_01_validate import run as validate
from .stage_02_typology import run as compute_typology
from .stage_03_classify import run as classify
from .stage_04_manifest import run as generate_manifest
from .stage_05_diagnostic import run as run_diagnostic

__all__ = [
    'validate',
    'compute_typology',
    'classify',
    'generate_manifest',
    'run_diagnostic',
]
