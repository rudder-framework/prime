"""Engine: level_histogram — distribution shape analysis."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'level_histogram_uniformity': np.nan, 'level_histogram_concentration': np.nan,
           'level_histogram_peak_ratio': np.nan, 'level_histogram_tail_weight': np.nan,
           'level_histogram_bimodality': np.nan}
    n = len(y)
    if n < 10:
        return nan
    n_bins = min(20, max(5, int(np.sqrt(n))))
    counts, _ = np.histogram(y, bins=n_bins)
    probs = counts / n
    uniform = 1.0 / n_bins
    uniformity = float(1.0 - np.sum(np.abs(probs - uniform)))
    concentration = float(np.max(probs))
    sorted_p = np.sort(probs)[::-1]
    peak_ratio = float(sorted_p[0] / sorted_p[1]) if len(sorted_p) > 1 and sorted_p[1] > 1e-15 else np.nan
    # Tail weight: fraction of data in extreme bins
    tail_weight = float((counts[0] + counts[-1]) / n)
    # Bimodality: Sarle's coefficient
    m = np.mean(y)
    s = np.std(y, ddof=1)
    if s < 1e-15:
        return nan
    z = (y - m) / s
    skew2 = float(np.mean(z ** 3)) ** 2
    kurt = float(np.mean(z ** 4) - 3.0)
    bc = float((skew2 + 1) / (kurt + 3)) if (kurt + 3) > 1e-15 else 0.0
    return {
        'level_histogram_uniformity': uniformity,
        'level_histogram_concentration': concentration,
        'level_histogram_peak_ratio': peak_ratio,
        'level_histogram_tail_weight': tail_weight,
        'level_histogram_bimodality': bc,
    }
