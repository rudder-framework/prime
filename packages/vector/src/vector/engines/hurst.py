"""Engine: hurst — Hurst exponent and R/S R²."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'hurst_exponent': np.nan, 'hurst_r2': np.nan}
    if len(y) < 32:
        return nan

    try:
        from pmtvs import hurst_exponent, hurst_r2
        return {
            'hurst_exponent': float(hurst_exponent(y)),
            'hurst_r2': float(hurst_r2(y)),
        }
    except (ImportError, Exception):
        return nan
