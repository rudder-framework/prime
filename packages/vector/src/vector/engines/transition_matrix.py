"""Engine: transition_matrix — state transition statistics."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'transition_matrix_entropy': np.nan, 'transition_matrix_self_loop': np.nan,
           'transition_matrix_max_self_loop': np.nan, 'transition_matrix_asymmetry': np.nan,
           'transition_matrix_n_active': np.nan, 'transition_matrix_sparsity': np.nan}
    if len(y) < 20:
        return nan

    try:
        from pmtvs import transition_analysis
        result = transition_analysis(y)
        if not isinstance(result, dict):
            return nan
        return {
            'transition_matrix_entropy': float(result.get('entropy', np.nan)),
            'transition_matrix_self_loop': float(result.get('self_loop', np.nan)),
            'transition_matrix_max_self_loop': float(result.get('max_self_loop', np.nan)),
            'transition_matrix_asymmetry': float(result.get('asymmetry', np.nan)),
            'transition_matrix_n_active': float(result.get('n_active', np.nan)),
            'transition_matrix_sparsity': float(result.get('sparsity', np.nan)),
        }
    except (ImportError, Exception):
        return nan
