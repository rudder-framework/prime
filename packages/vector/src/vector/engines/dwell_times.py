"""Engine: dwell_times — time spent in discrete states."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'dwell_mean': np.nan, 'dwell_std': np.nan, 'dwell_max': np.nan,
           'dwell_min': np.nan, 'dwell_cv': np.nan, 'dwell_n': np.nan}
    if len(y) < 10:
        return nan

    try:
        from pmtvs import dwell_analysis
        result = dwell_analysis(y)
        if not isinstance(result, dict):
            return nan
        return {
            'dwell_mean': float(result.get('mean', np.nan)),
            'dwell_std': float(result.get('std', np.nan)),
            'dwell_max': float(result.get('max', np.nan)),
            'dwell_min': float(result.get('min', np.nan)),
            'dwell_cv': float(result.get('cv', np.nan)),
            'dwell_n': float(result.get('n', np.nan)),
        }
    except (ImportError, Exception):
        return nan
