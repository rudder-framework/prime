"""Engine: statistics — kurtosis, skewness, crest factor."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 4:
        return {'statistics_kurtosis': np.nan, 'statistics_skewness': np.nan,
                'statistics_crest_factor': np.nan}

    m = np.mean(y)
    s = np.std(y, ddof=1)

    if s < 1e-15:
        return {'statistics_kurtosis': 0.0, 'statistics_skewness': 0.0,
                'statistics_crest_factor': 0.0}

    z = (y - m) / s
    kurt = float(np.mean(z ** 4) - 3.0)
    skew = float(np.mean(z ** 3))
    rms_val = np.sqrt(np.mean(y ** 2))
    crest = float(np.max(np.abs(y)) / rms_val) if rms_val > 1e-15 else 0.0

    return {
        'statistics_kurtosis': kurt,
        'statistics_skewness': skew,
        'statistics_crest_factor': crest,
    }
