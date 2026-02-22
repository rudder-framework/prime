"""Engine: entropy — Shannon, normalized, conditional, rate, excess."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    nan = {'entropy_shannon': np.nan, 'entropy_normalized': np.nan,
           'entropy_conditional': np.nan, 'entropy_rate': np.nan,
           'entropy_excess': np.nan}
    if len(y) < 4:
        return nan

    try:
        from pmtvs import shannon_entropy
        se = float(shannon_entropy(y, n_bins))
    except (ImportError, Exception):
        return nan

    max_entropy = np.log(n_bins)
    norm = se / max_entropy if max_entropy > 0 else 0.0

    return {
        'entropy_shannon': se,
        'entropy_normalized': norm,
        'entropy_conditional': np.nan,  # requires joint distribution
        'entropy_rate': np.nan,         # requires sequential computation
        'entropy_excess': np.nan,       # requires block entropy
    }
