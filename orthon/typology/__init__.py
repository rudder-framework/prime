"""ORTHON Typology Classification Module."""

__all__ = []

# Level 2 Corrections (PR4) - may fail if orthon.config is broken
try:
    from .level2_corrections import (
        is_first_bin_artifact,
        is_genuine_periodic,
        classify_temporal_pattern,
        classify_spectral,
        correct_engines,
        correct_visualizations,
        apply_corrections,
    )
    __all__.extend([
        'is_first_bin_artifact',
        'is_genuine_periodic',
        'classify_temporal_pattern',
        'classify_spectral',
        'correct_engines',
        'correct_visualizations',
        'apply_corrections',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import level2_corrections: {e}")

# Discrete/Sparse Classification (PR5)
try:
    from .discrete_sparse import (
        is_constant,
        is_binary,
        is_discrete,
        is_impulsive,
        is_event,
        is_step,
        is_intermittent,
        classify_discrete_sparse,
        apply_discrete_sparse_classification,
        get_spectral_for_discrete,
        get_engines_for_discrete,
    )
    __all__.extend([
        'is_constant',
        'is_binary',
        'is_discrete',
        'is_impulsive',
        'is_event',
        'is_step',
        'is_intermittent',
        'classify_discrete_sparse',
        'apply_discrete_sparse_classification',
        'get_spectral_for_discrete',
        'get_engines_for_discrete',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import discrete_sparse: {e}")

# Robust CONSTANT Detection (PR8)
try:
    from .constant_detection import (
        is_constant_signal,
        classify_constant_from_row,
        validate_constant_detection,
        CONSTANT_CONFIG,
    )
    __all__.extend([
        'is_constant_signal',
        'classify_constant_from_row',
        'validate_constant_detection',
        'CONSTANT_CONFIG',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import constant_detection: {e}")

# Window Factor - standalone, should always work
from .window_factor import (
    compute_window_factor,
    add_window_factor,
)
__all__.extend([
    'compute_window_factor',
    'add_window_factor',
])
