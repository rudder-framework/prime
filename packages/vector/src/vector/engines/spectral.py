"""Engine: spectral — slope, dominant freq, entropy, centroid, bandwidth."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'spectral_slope': np.nan, 'spectral_dominant_freq': np.nan,
           'spectral_entropy': np.nan, 'spectral_centroid': np.nan,
           'spectral_bandwidth': np.nan}
    if len(y) < 16:
        return nan

    try:
        from pmtvs import (spectral_slope, dominant_frequency,
                           spectral_entropy, spectral_centroid,
                           spectral_bandwidth)
        return {
            'spectral_slope': float(spectral_slope(y)),
            'spectral_dominant_freq': float(dominant_frequency(y)),
            'spectral_entropy': float(spectral_entropy(y)),
            'spectral_centroid': float(spectral_centroid(y)),
            'spectral_bandwidth': float(spectral_bandwidth(y)),
        }
    except (ImportError, Exception):
        return nan
