"""
vector — Windowed Feature Extraction
=====================================

Three scales, one package:

    vector.signal.compute_signal(signal_id, values, window_size, stride)
        → List of dicts, one per window, all features namespaced.

    vector.cohort.compute_cohort(signal_matrix, cohort_id, window_index, feature_names)
        → Dict with centroid per feature + dispersion metrics.

    vector.system.compute_system(cohort_matrix, window_index, feature_names)
        → Dict with system centroid + fleet dispersion.

Engine registry:
    vector.registry.get_registry()
        → Registry with 44 engines, YAML-configured, lazily loaded.

All engine output keys are namespaced: {engine}_{key}. Zero collisions.
"""

__version__ = '0.1.0'

from vector.registry import get_registry, Registry
from vector import signal
from vector import cohort
from vector import system
