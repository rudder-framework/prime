"""Engine: cycle_counting — rainflow-style cycle analysis."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {k: np.nan for k in [
        'cycle_counting_n_cycles', 'cycle_counting_n_full',
        'cycle_counting_n_half', 'cycle_counting_max_range',
        'cycle_counting_mean_range', 'cycle_counting_accumulation']}
    n = len(y)
    if n < 4:
        return nan
    # Simple peak-valley cycle counting
    d = np.diff(y)
    sign_changes = np.where(d[:-1] * d[1:] < 0)[0] + 1
    if len(sign_changes) < 2:
        return {**nan, 'cycle_counting_n_cycles': 0.0, 'cycle_counting_n_full': 0.0,
                'cycle_counting_n_half': 0.0}
    extrema = y[sign_changes]
    ranges = np.abs(np.diff(extrema))
    n_half = len(ranges)
    n_full = n_half // 2
    return {
        'cycle_counting_n_cycles': float(n_full + 0.5 * (n_half % 2)),
        'cycle_counting_n_full': float(n_full),
        'cycle_counting_n_half': float(n_half),
        'cycle_counting_max_range': float(np.max(ranges)) if len(ranges) > 0 else 0.0,
        'cycle_counting_mean_range': float(np.mean(ranges)) if len(ranges) > 0 else 0.0,
        'cycle_counting_accumulation': float(np.sum(ranges)),
    }
