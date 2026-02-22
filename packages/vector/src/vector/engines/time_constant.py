"""Engine: time_constant — exponential fit tau, equilibrium, R²."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'time_constant_tau': np.nan, 'time_constant_equilibrium': np.nan,
           'time_constant_fit_r2': np.nan, 'time_constant_is_decay': np.nan}
    if len(y) < 10:
        return nan

    try:
        from pmtvs import time_constant
        result = time_constant(y)
        if not isinstance(result, dict):
            return nan
        return {
            'time_constant_tau': float(result.get('tau', np.nan)),
            'time_constant_equilibrium': float(result.get('equilibrium', np.nan)),
            'time_constant_fit_r2': float(result.get('fit_r2', np.nan)),
            'time_constant_is_decay': float(result.get('is_decay', np.nan)),
        }
    except (ImportError, Exception):
        return nan
