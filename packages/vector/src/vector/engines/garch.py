"""Engine: garch — GARCH(1,1) volatility model parameters."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'garch_omega': np.nan, 'garch_alpha': np.nan,
           'garch_beta': np.nan, 'garch_persistence': np.nan}
    if len(y) < 50:
        return nan
    try:
        from pmtvs import garch_fit
        result = garch_fit(y)
        if isinstance(result, dict):
            omega = float(result.get('omega', np.nan))
            alpha = float(result.get('alpha', np.nan))
            beta = float(result.get('beta', np.nan))
            return {'garch_omega': omega, 'garch_alpha': alpha,
                    'garch_beta': beta, 'garch_persistence': alpha + beta}
        return nan
    except (ImportError, Exception):
        return nan
