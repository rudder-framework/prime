"""Engine: recurrence_rate — standalone recurrence quantification."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 50:
        return {'recurrence_rate_value': np.nan}
    try:
        from pmtvs import recurrence_rate
        return {'recurrence_rate_value': float(recurrence_rate(y))}
    except ImportError:
        n = min(len(y), 500)
        sub = y[:n]
        threshold = 0.1 * np.std(sub)
        dist = np.abs(sub[:, None] - sub[None, :])
        return {'recurrence_rate_value': float(np.sum(dist < threshold) / (n * n))}
