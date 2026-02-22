"""Engine: rate_of_change — derivative statistics."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 2:
        return {k: np.nan for k in [
            'rate_of_change_mean', 'rate_of_change_max', 'rate_of_change_min',
            'rate_of_change_std', 'rate_of_change_abs_max']}
    d = np.diff(y)
    return {
        'rate_of_change_mean': float(np.mean(d)),
        'rate_of_change_max': float(np.max(d)),
        'rate_of_change_min': float(np.min(d)),
        'rate_of_change_std': float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
        'rate_of_change_abs_max': float(np.max(np.abs(d))),
    }
