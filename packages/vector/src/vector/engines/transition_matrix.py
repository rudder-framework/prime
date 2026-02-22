"""Engine: transition_matrix — state transition statistics."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'transition_matrix_entropy': np.nan, 'transition_matrix_self_loop': np.nan,
           'transition_matrix_max_self_loop': np.nan, 'transition_matrix_asymmetry': np.nan,
           'transition_matrix_n_active': np.nan, 'transition_matrix_sparsity': np.nan}
    n = len(y)
    if n < 20:
        return nan
    # Quantize to states
    n_states = min(5, max(2, int(np.sqrt(n) / 5)))
    states = np.digitize(y, np.linspace(np.min(y), np.max(y), n_states + 1)) - 1
    states = np.clip(states, 0, n_states - 1)
    # Build transition matrix
    tm = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        tm[states[i], states[i + 1]] += 1
    row_sums = tm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tm_norm = tm / row_sums
    # Metrics
    diag = np.diag(tm_norm)
    self_loop = float(np.mean(diag))
    max_self = float(np.max(diag))
    asym = float(np.sum(np.abs(tm_norm - tm_norm.T)) / 2)
    nonzero = np.sum(tm > 0)
    total = n_states * n_states
    sparsity = float(1.0 - nonzero / total)
    # Entropy of transition probs
    flat = tm_norm.ravel()
    flat = flat[flat > 1e-15]
    entropy = float(-np.sum(flat * np.log(flat)))
    return {
        'transition_matrix_entropy': entropy,
        'transition_matrix_self_loop': self_loop,
        'transition_matrix_max_self_loop': max_self,
        'transition_matrix_asymmetry': asym,
        'transition_matrix_n_active': float(nonzero),
        'transition_matrix_sparsity': sparsity,
    }
