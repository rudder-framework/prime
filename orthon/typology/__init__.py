"""ORTHON Typology Classification Module."""

from .level2_corrections import (
    is_first_bin_artifact,
    is_genuine_periodic,
    classify_temporal_pattern,
    classify_spectral,
    correct_engines,
    correct_visualizations,
    apply_corrections,
)
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

__all__ = [
    # Level 2 Corrections (PR4)
    'is_first_bin_artifact',
    'is_genuine_periodic',
    'classify_temporal_pattern',
    'classify_spectral',
    'correct_engines',
    'correct_visualizations',
    'apply_corrections',
    # Discrete/Sparse Classification (PR5)
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
]
