"""
PR10: Multi-scale Representation
"""

from .characteristic_time import (
    compute_characteristic_time,
    classify_dynamics_speed,
    compute_window_stride,
    compute_window_config,
    add_window_columns,
    WINDOW_CONFIG,
)

from .system_window import (
    compute_system_window,
    classify_representation,
    compute_signal_representation,
    compute_system_representation,
    summarize_representations,
    REPRESENTATION_CONFIG,
)

from .manifest_generator import (
    generate_manifest,
    get_engines_for_type,
    manifest_to_yaml,
    save_manifest,
    BASE_ENGINES,
    ENGINE_ADDITIONS,
)

__all__ = [
    # Characteristic time (PR9)
    'compute_characteristic_time',
    'classify_dynamics_speed',
    'compute_window_stride',
    'compute_window_config',
    'add_window_columns',
    'WINDOW_CONFIG',
    # System window (PR10)
    'compute_system_window',
    'classify_representation',
    'compute_signal_representation',
    'compute_system_representation',
    'summarize_representations',
    'REPRESENTATION_CONFIG',
    # Manifest
    'generate_manifest',
    'get_engines_for_type',
    'manifest_to_yaml',
    'save_manifest',
    'BASE_ENGINES',
    'ENGINE_ADDITIONS',
]
