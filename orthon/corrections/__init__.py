"""
ORTHON Typology Classification Corrections

Post-processing corrections for typology classification.
Call these after compute_raw_measures() to fix known issues.

Usage:
    from orthon.corrections import apply_corrections, apply_level1_corrections

    row = compute_raw_measures(signal)
    row = apply_level1_corrections(row, signal_values)  # Optional stationarity fix
    row = apply_corrections(row)                         # Level 2 fixes
"""

from .level1_corrections import (
    detect_deterministic_trend,
    correct_stationarity,
    apply_level1_corrections,
)

from .level2_corrections import (
    is_first_bin_artifact,
    compute_corrected_dominant_freq,
    is_genuine_periodic,
    classify_temporal_pattern,
    classify_spectral,
    correct_visualizations,
    correct_engines,
    apply_corrections,
)

from .manifest_corrections import compute_global_stride_default

__all__ = [
    # Level 1
    'detect_deterministic_trend',
    'correct_stationarity',
    'apply_level1_corrections',
    # Level 2
    'is_first_bin_artifact',
    'compute_corrected_dominant_freq',
    'is_genuine_periodic',
    'classify_temporal_pattern',
    'classify_spectral',
    'correct_visualizations',
    'correct_engines',
    'apply_corrections',
    # Manifest
    'compute_global_stride_default',
]
