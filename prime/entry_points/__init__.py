"""
Prime Entry Points - Ordered Pipeline Stages
==============================================

Thin orchestrators that call engines/modules for computation.
Entry points do NOT contain compute logic - only orchestration.

Pipeline Order (Pre-Manifold):
    stage_01_validate   → Validation (remove constants, duplicates)
    stage_02_typology   → Compute raw typology measures (27 metrics)
    stage_03_classify   → Apply classification (discrete/sparse → continuous)
    stage_04_manifest   → Generate manifest for Manifold
    stage_05_diagnostic → Run diagnostic assessment (uses engines)

Post-Manifold / Support:
    stage_06_interpret  → Interpret Manifold outputs (dynamics + physics)
    stage_07_predict    → Predict RUL, health, anomalies
    stage_08_alert      → Early warning / failure fingerprints
    stage_09_explore    → Manifold visualization
    stage_10_inspect    → File inspection / capability detection
    stage_11_fetch      → Read, profile, and validate raw data
    stage_12_stream     → Real-time streaming analysis
    stage_13_train      → Train ML models on Manifold features

Usage:
    python -m prime.entry_points.stage_01_validate observations.parquet -o validated.parquet
    python -m prime.entry_points.stage_02_typology observations.parquet -o typology_raw.parquet
    python -m prime.entry_points.stage_03_classify typology_raw.parquet -o typology.parquet
    python -m prime.entry_points.stage_04_manifest typology.parquet -o manifest.yaml
    python -m prime.entry_points.stage_05_diagnostic observations.parquet -o report.txt
    python -m prime.entry_points.stage_06_interpret /path/to/manifold/output
    python -m prime.entry_points.stage_07_predict /path/to/manifold/output --mode health
    python -m prime.entry_points.stage_08_alert observations.parquet
    python -m prime.entry_points.stage_09_explore /path/to/manifold/output
    python -m prime.entry_points.stage_10_inspect data.parquet
    python -m prime.entry_points.stage_11_fetch raw_data.csv -o observations.parquet
    python -m prime.entry_points.stage_12_stream dashboard --source turbofan
    python -m prime.entry_points.stage_13_train --model xgboost

Manifold computes numbers. Prime classifies.
"""

# Import run functions for convenience — Pre-Manifold pipeline
from .stage_01_validate import run as validate
from .stage_02_typology import run as compute_typology
from .stage_03_classify import run as classify
from .stage_04_manifest import run as generate_manifest
from .stage_05_diagnostic import run as run_diagnostic

# Post-Manifold and support entry points — lazy imports to avoid
# heavy dependencies at package load time
def __getattr__(name):
    _lazy = {
        'interpret': ('.stage_06_interpret', 'run'),
        'predict': ('.stage_07_predict', 'run'),
        'alert': ('.stage_08_alert', 'run'),
        'explore': ('.stage_09_explore', 'run'),
        'inspect': ('.stage_10_inspect', 'run'),
        'fetch': ('.stage_11_fetch', 'run'),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Pre-Manifold pipeline
    'validate',
    'compute_typology',
    'classify',
    'generate_manifest',
    'run_diagnostic',
    # Post-Manifold (lazy)
    'interpret',
    'predict',
    'alert',
    'explore',
    'inspect',
    'fetch',
    # stream and train have heavy deps — import directly:
    # from prime.entry_points.stage_12_stream import run as stream
    # from prime.entry_points.stage_13_train import run as train
]
