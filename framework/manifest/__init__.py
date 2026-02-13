"""
Manifest generation: window sizing, engine selection, output configuration.

RUDDER Manifest Generator v2.5 - Per-Engine Window Specification
"""

from .generator import (
    ENGINE_ADJUSTMENTS,
    VIZ_ADJUSTMENTS,
    BASE_ENGINES,
    BASE_VISUALIZATIONS,
    ENGINE_MIN_WINDOWS,
    apply_engine_adjustments,
    apply_viz_adjustments,
    get_window_params,
    get_output_hints,
    build_signal_config,
    build_manifest,
    build_atlas_config,
    validate_manifest,
    manifest_to_yaml,
    save_manifest,
    compute_engine_window_overrides,
)

from .window_recommender import recommend_window
from .system_window import compute_system_window
from .characteristic_time import compute_characteristic_time
from .domain_clock import DomainClock

__all__ = [
    # Generator
    'ENGINE_ADJUSTMENTS',
    'VIZ_ADJUSTMENTS',
    'BASE_ENGINES',
    'BASE_VISUALIZATIONS',
    'ENGINE_MIN_WINDOWS',
    'apply_engine_adjustments',
    'apply_viz_adjustments',
    'get_window_params',
    'get_output_hints',
    'build_signal_config',
    'build_manifest',
    'build_atlas_config',
    'validate_manifest',
    'manifest_to_yaml',
    'save_manifest',
    'compute_engine_window_overrides',
    # Window
    'recommend_window',
    'compute_system_window',
    'compute_characteristic_time',
    'DomainClock',
]
