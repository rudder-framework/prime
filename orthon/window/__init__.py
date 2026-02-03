"""
PR9: Window/Stride from Characteristic Time
"""

from .characteristic_time import (
    compute_characteristic_time,
    classify_dynamics_speed,
    compute_window_stride,
    compute_window_config,
    add_window_columns,
    WINDOW_CONFIG,
)

from .manifest_generator import (
    build_signal_config,
    generate_manifest,
    get_engines_for_type,
    manifest_to_yaml,
    save_manifest,
    BASE_ENGINES,
    ENGINE_ADDITIONS,
)

__all__ = [
    # Characteristic time
    'compute_characteristic_time',
    'classify_dynamics_speed',
    'compute_window_stride',
    'compute_window_config',
    'add_window_columns',
    'WINDOW_CONFIG',
    # Manifest
    'build_signal_config',
    'generate_manifest',
    'get_engines_for_type',
    'manifest_to_yaml',
    'save_manifest',
    'BASE_ENGINES',
    'ENGINE_ADDITIONS',
]
