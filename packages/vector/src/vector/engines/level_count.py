"""Engine: level_count — discrete level analysis."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'level_count_n': np.nan, 'level_count_occupancy': np.nan,
           'level_count_dominant': np.nan, 'level_count_spread': np.nan,
           'level_count_entropy': np.nan}
    n = len(y)
    if n < 4:
        return nan
    n_bins = min(20, max(3, int(np.sqrt(n))))
    counts, edges = np.histogram(y, bins=n_bins)
    nonzero = counts[counts > 0]
    n_levels = len(nonzero)
    probs = nonzero / n
    dom_idx = int(np.argmax(counts))
    dominant = float((edges[dom_idx] + edges[dom_idx + 1]) / 2)
    occupancy = float(np.max(counts) / n)
    spread = float(np.std(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-30)))
    return {
        'level_count_n': float(n_levels),
        'level_count_occupancy': occupancy,
        'level_count_dominant': dominant,
        'level_count_spread': spread,
        'level_count_entropy': entropy,
    }
