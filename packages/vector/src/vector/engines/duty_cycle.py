"""Engine: duty_cycle — time spent above/below thresholds."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'duty_cycle_dominant': np.nan, 'duty_cycle_secondary': np.nan,
           'duty_cycle_ratio': np.nan, 'duty_cycle_balance': np.nan,
           'duty_cycle_range': np.nan}
    n = len(y)
    if n < 4:
        return nan
    median = np.median(y)
    above = np.sum(y > median) / n
    below = 1.0 - above
    dominant = max(above, below)
    secondary = min(above, below)
    ratio = dominant / secondary if secondary > 1e-10 else np.nan
    return {
        'duty_cycle_dominant': float(dominant),
        'duty_cycle_secondary': float(secondary),
        'duty_cycle_ratio': ratio,
        'duty_cycle_balance': float(1.0 - abs(above - below)),
        'duty_cycle_range': float(np.max(y) - np.min(y)),
    }
