"""Engine: complexity — sample, permutation, approximate entropy."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'complexity_sample_entropy': np.nan,
           'complexity_permutation_entropy': np.nan,
           'complexity_approximate_entropy': np.nan}
    if n < 30:
        return nan

    try:
        from pmtvs import sample_entropy, permutation_entropy, approximate_entropy
        se = float(sample_entropy(y))
        pe = float(permutation_entropy(y))
        ae = float(approximate_entropy(y))
    except ImportError:
        se = np.nan
        pe = _perm_entropy(y)
        ae = np.nan

    return {
        'complexity_sample_entropy': se,
        'complexity_permutation_entropy': pe,
        'complexity_approximate_entropy': ae,
    }


def _perm_entropy(x, order=3):
    """Fallback permutation entropy."""
    from math import factorial
    n = len(x)
    if n < order + 1:
        return np.nan
    counts = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i + order]))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    ent = -np.sum(probs * np.log(probs + 1e-30))
    return float(ent / np.log(factorial(order)))
