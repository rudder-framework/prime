"""Engine: variance_growth — rate and ratio of variance expansion."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 20:
        return {'variance_growth_rate': np.nan, 'variance_growth_ratio': np.nan}
    half = n // 2
    var1 = float(np.var(y[:half]))
    var2 = float(np.var(y[half:]))
    ratio = var2 / var1 if var1 > 1e-15 else np.nan
    rate = (var2 - var1) / half if half > 0 else np.nan
    return {'variance_growth_rate': rate, 'variance_growth_ratio': ratio}
