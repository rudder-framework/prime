"""Engine: cycle_counting — rainflow-style cycle analysis."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {k: np.nan for k in [
        'cycle_counting_n_cycles', 'cycle_counting_n_full',
        'cycle_counting_n_half', 'cycle_counting_max_range',
        'cycle_counting_mean_range', 'cycle_counting_accumulation']}
    if len(y) < 4:
        return nan

    try:
        from pmtvs import cycle_counting
        result = cycle_counting(y)
        if not isinstance(result, dict):
            return nan
        return {
            'cycle_counting_n_cycles': float(result.get('n_cycles', np.nan)),
            'cycle_counting_n_full': float(result.get('n_full', np.nan)),
            'cycle_counting_n_half': float(result.get('n_half', np.nan)),
            'cycle_counting_max_range': float(result.get('max_range', 0.0)),
            'cycle_counting_mean_range': float(result.get('mean_range', 0.0)),
            'cycle_counting_accumulation': float(result.get('accumulation', np.nan)),
        }
    except (ImportError, Exception):
        return nan
