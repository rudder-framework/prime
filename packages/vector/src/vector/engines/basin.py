"""Engine: basin — basin stability, transition probability, n_attractors."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'basin_stability': np.nan, 'basin_transition_prob': np.nan,
           'basin_n_attractors': np.nan}
    if len(y) < 50:
        return nan
    try:
        from pmtvs import basin_stability
        result = basin_stability(y)
        if isinstance(result, dict):
            return {
                'basin_stability': float(result.get('basin_stability', np.nan)),
                'basin_transition_prob': float(result.get('transition_prob', np.nan)),
                'basin_n_attractors': float(result.get('n_attractors', np.nan)),
            }
        return nan
    except (ImportError, Exception):
        return nan
