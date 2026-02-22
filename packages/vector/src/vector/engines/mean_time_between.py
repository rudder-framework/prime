"""Engine: mean_time_between — interval statistics for threshold crossings."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'mtb_mean': np.nan, 'mtb_std': np.nan, 'mtb_max': np.nan,
           'mtb_min': np.nan, 'mtb_cv': np.nan}
    n = len(y)
    if n < 10:
        return nan
    median = np.median(y)
    above = y > median
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    if len(crossings) < 2:
        return nan
    intervals = np.diff(crossings).astype(float)
    m = float(np.mean(intervals))
    s = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
    return {
        'mtb_mean': m, 'mtb_std': s,
        'mtb_max': float(np.max(intervals)),
        'mtb_min': float(np.min(intervals)),
        'mtb_cv': s / m if m > 1e-15 else np.nan,
    }
