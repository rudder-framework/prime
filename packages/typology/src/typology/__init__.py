"""
typology — Signal Classification & Window Sizing
=================================================

Three entry points:

    typology.from_observations(values)
        Called by Manifold BEFORE windowing.
        Cheap O(n) measures → window recommendation + initial classification.

    typology.from_features(features_dict)
        Called by Manifold per-window and by Prime per-signal.
        Pure threshold logic → 10-dimension classification. Microseconds.

    typology.window_from_length(n_samples)
        Called by Manifold when no measures exist yet.
        Pure defaults from observation count. Nanoseconds.

Usage:
    import typology

    # Before any computation — just need window size
    ws = typology.window_from_length(14000)
    # → {'window_size': 256, 'stride': 64, 'source': 'length_default'}

    # From raw observations — window + initial class
    result = typology.from_observations(values)
    # → {'measures': {...}, 'window': {...}, 'classification': {...}}

    # From pre-computed features — full 10-dimension classification
    classes = typology.from_features({'hurst': 0.92, 'perm_entropy': 0.3, ...})
    # → {'temporal': 'TRENDING', 'memory': 'LONG_MEMORY', ...}
"""

__version__ = '0.1.0'

import numpy as np
from typing import Dict, Any

from typology.classify import classify
from typology.observe import observe
from typology.window import from_length, from_measures, system_window
from typology.config import CONFIG, get as get_config


def from_observations(values: np.ndarray) -> Dict[str, Any]:
    """
    Full pipeline from raw signal values.
    Computes cheap measures, derives window, classifies.

    Args:
        values: 1D array of signal observations, sorted by index.

    Returns:
        {
            'measures': {...},           # all cheap O(n) measures
            'window': {...},             # window_size, stride, source
            'classification': {...},     # 10-dimension classification
        }
    """
    measures = observe(values)

    if measures['is_constant']:
        window = {'window_size': measures['n_samples'],
                  'stride': measures['n_samples'],
                  'source': 'constant'}
    else:
        window = from_measures(measures)

    classification = classify(measures)

    return {
        'measures': measures,
        'window': window,
        'classification': classification,
    }


def from_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify from pre-computed features.
    Called per-window by Manifold, per-signal by Prime.

    Args:
        features: Dict of pre-computed measures (from signal_vector or
                  typology_raw). See classify.classify() for accepted keys.

    Returns:
        10-dimension classification dict.
    """
    return classify(features)


def window_from_length(n_samples: int) -> Dict[str, Any]:
    """
    Default window/stride from observation count alone.
    No measures needed. Called before any computation.

    Args:
        n_samples: Number of observations.

    Returns:
        {window_size, stride, source}
    """
    return from_length(n_samples)
