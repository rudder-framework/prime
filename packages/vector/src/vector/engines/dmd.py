"""Engine: dmd — Dynamic Mode Decomposition."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'dmd_dominant_freq': np.nan, 'dmd_growth_rate': np.nan,
           'dmd_is_stable': np.nan, 'dmd_n_modes': np.nan}
    if len(y) < 20:
        return nan
    try:
        from pmtvs import dmd_decompose
        result = dmd_decompose(y)
        if isinstance(result, dict):
            return {
                'dmd_dominant_freq': float(result.get('dominant_freq', np.nan)),
                'dmd_growth_rate': float(result.get('growth_rate', np.nan)),
                'dmd_is_stable': float(result.get('is_stable', np.nan)),
                'dmd_n_modes': float(result.get('n_modes', np.nan)),
            }
        return nan
    except (ImportError, Exception):
        return nan
