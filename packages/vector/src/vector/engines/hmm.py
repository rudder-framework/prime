"""Engine: hmm — Hidden Markov Model fit."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'hmm_n_states': np.nan, 'hmm_current_state': np.nan,
           'hmm_current_state_prob': np.nan, 'hmm_state_entropy': np.nan,
           'hmm_transition_rate': np.nan, 'hmm_bic': np.nan,
           'hmm_log_likelihood': np.nan}
    if len(y) < 50:
        return nan
    try:
        from pmtvs import hmm_fit
        result = hmm_fit(y)
        if isinstance(result, dict):
            return {
                'hmm_n_states': float(result.get('n_states', np.nan)),
                'hmm_current_state': float(result.get('current_state', np.nan)),
                'hmm_current_state_prob': float(result.get('current_state_prob', np.nan)),
                'hmm_state_entropy': float(result.get('state_entropy', np.nan)),
                'hmm_transition_rate': float(result.get('transition_rate', np.nan)),
                'hmm_bic': float(result.get('bic', np.nan)),
                'hmm_log_likelihood': float(result.get('log_likelihood', np.nan)),
            }
        return nan
    except (ImportError, Exception):
        return nan
