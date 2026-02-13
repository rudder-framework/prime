"""
Prime Analysis Layer

Reads Manifold's parquet outputs and interprets them.
Manifold computes. Prime classifies, analyzes, and reports.

Modules:
    twenty_twenty                — 20/20 early detection (geometric deformation vs lifecycle)
    canary                       — Canary signal identification (which sensor drives collapse)
    thermodynamics               — Thermodynamic interpretation of eigenvalue spectrum
    window_optimization          — Window/stride grid sweep (fast raw eigendecomp)
    window_optimization_manifold — Window/stride grid sweep (full Manifold pipeline)
    study                        — Study runner (orchestrates all analyses on a dataset)

Usage:
    python -m prime.analysis.study --data path/to/dataset_dir
    python -m prime.analysis.twenty_twenty --geometry state_geometry.parquet --observations observations.parquet
    python -m prime.analysis.canary --signal-vector signal_vector.parquet --geometry state_geometry.parquet --observations observations.parquet
    python -m prime.analysis.thermodynamics --geometry state_geometry.parquet
    python -m prime.analysis.window_optimization --observations observations.parquet --typology typology.parquet
    python -m prime.analysis.window_optimization_manifold --data-dir path/to/FD001
"""

from .twenty_twenty import run_twenty_twenty, get_lifecycle_per_cohort
from .canary import analyze_signal_velocity, analyze_signal_collapse_correlation, analyze_single_signal_rul
from .thermodynamics import compute_thermodynamics, detect_phase_transitions
from .window_optimization import run_grid_sweep, fast_eigendecomp, compute_effdim_trajectories
from .study import run_study

__all__ = [
    'run_twenty_twenty',
    'get_lifecycle_per_cohort',
    'analyze_signal_velocity',
    'analyze_signal_collapse_correlation',
    'analyze_single_signal_rul',
    'compute_thermodynamics',
    'detect_phase_transitions',
    'run_grid_sweep',
    'fast_eigendecomp',
    'compute_effdim_trajectories',
    'run_study',
]
