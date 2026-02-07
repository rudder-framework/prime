"""
ORTHON Entry Points - Ordered Pipeline Stages
==============================================

Thin orchestrators that call engines/modules for computation.
Entry points do NOT contain compute logic - only orchestration.

Pipeline Order (pre-PRISM):
    stage_01_validate   → Validation (remove constants, duplicates)
    stage_02_typology   → Compute raw typology measures (27 metrics)
    stage_03_classify   → Apply classification (discrete/sparse → continuous)
    stage_04_manifest   → Generate manifest for PRISM
    stage_05_diagnostic → Run diagnostic assessment (uses engines)

Post-PRISM / Support:
    stage_06_interpret  → Interpret PRISM outputs (dynamics + physics)
    stage_07_predict    → Predict RUL, health, anomalies
    stage_08_alert      → Early warning / failure fingerprints
    stage_09_explore    → Manifold visualization
    stage_10_inspect    → File inspection / capability detection
    stage_11_fetch      → Read, profile, and validate raw data
    stage_12_stream     → Real-time streaming analysis
    stage_13_train      → Train ML models on PRISM features

Usage:
    python -m orthon.entry_points.stage_01_validate observations.parquet -o validated.parquet
    python -m orthon.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
    python -m orthon.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
    python -m orthon.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
    python -m orthon.entry_points.stage_05_diagnostic observations.parquet -o report.txt
    python -m orthon.entry_points.stage_06_interpret /path/to/prism/output
    python -m orthon.entry_points.stage_07_predict /path/to/prism/output --mode health
    python -m orthon.entry_points.stage_08_alert observations.parquet
    python -m orthon.entry_points.stage_09_explore /path/to/prism/output
    python -m orthon.entry_points.stage_10_inspect data.parquet
    python -m orthon.entry_points.stage_11_fetch raw_data.csv -o observations.parquet
    python -m orthon.entry_points.stage_12_stream dashboard --source turbofan
    python -m orthon.entry_points.stage_13_train --model xgboost

PRISM computes numbers. ORTHON classifies.
"""

# Import run functions for convenience — pre-PRISM pipeline
from .stage_01_validate import run as validate
from .stage_02_typology import run as compute_typology
from .stage_03_classify import run as classify
from .stage_04_manifest import run as generate_manifest
from .stage_05_diagnostic import run as run_diagnostic

# Post-PRISM and support entry points — lazy imports to avoid
# heavy dependencies at package load time
from .stage_06_interpret import run as interpret
from .stage_07_predict import run as predict
from .stage_08_alert import run as alert
from .stage_09_explore import run as explore
from .stage_10_inspect import run as inspect
from .stage_11_fetch import run as fetch

__all__ = [
    # Pre-PRISM pipeline
    'validate',
    'compute_typology',
    'classify',
    'generate_manifest',
    'run_diagnostic',
    # Post-PRISM
    'interpret',
    'predict',
    'alert',
    'explore',
    'inspect',
    'fetch',
    # stream and train have heavy deps — import directly:
    # from orthon.entry_points.stage_12_stream import run as stream
    # from orthon.entry_points.stage_13_train import run as train
]
