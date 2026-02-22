"""Engine: lof — Local Outlier Factor statistics."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'lof_max': np.nan, 'lof_mean': np.nan, 'lof_std': np.nan,
           'lof_outlier_fraction': np.nan, 'lof_n_outliers': np.nan}
    n = len(y)
    if n < 20:
        return nan

    try:
        from pmtvs import local_outlier_factor
        result = local_outlier_factor(y)
        if not isinstance(result, dict):
            return nan
        scores = result.get('scores')
        if scores is None or len(scores) == 0:
            return nan
        outliers = scores > 1.5
        return {
            'lof_max': float(np.max(scores)),
            'lof_mean': float(np.mean(scores)),
            'lof_std': float(np.std(scores)),
            'lof_outlier_fraction': float(np.sum(outliers) / n),
            'lof_n_outliers': float(np.sum(outliers)),
        }
    except (ImportError, Exception):
        return nan
