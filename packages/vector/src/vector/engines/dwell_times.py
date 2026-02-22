"""Engine: dwell_times — time spent in discrete states."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'dwell_mean': np.nan, 'dwell_std': np.nan, 'dwell_max': np.nan,
           'dwell_min': np.nan, 'dwell_cv': np.nan, 'dwell_n': np.nan}
    n = len(y)
    if n < 10:
        return nan
    # Quantize to bins
    n_bins = min(10, int(np.sqrt(n)))
    bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins + 1))
    # Find dwell times (consecutive same bin)
    dwells = []
    current_len = 1
    for i in range(1, len(bins)):
        if bins[i] == bins[i - 1]:
            current_len += 1
        else:
            dwells.append(current_len)
            current_len = 1
    dwells.append(current_len)
    d = np.array(dwells, dtype=float)
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return {
        'dwell_mean': m, 'dwell_std': s,
        'dwell_max': float(np.max(d)),
        'dwell_min': float(np.min(d)),
        'dwell_cv': s / m if m > 1e-15 else 0.0,
        'dwell_n': float(len(d)),
    }
