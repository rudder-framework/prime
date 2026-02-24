"""
Manifest generation: window sizing, engine selection, output configuration.

Manifest Generator v2.5 - Per-Engine Window Specification
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
]
