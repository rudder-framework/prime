"""Engine: memory — Hurst and R² via memory-specific estimation."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 32:
        return {'memory_hurst': np.nan, 'memory_r2': np.nan}
    try:
        from pmtvs import hurst_exponent, hurst_r2
        return {'memory_hurst': float(hurst_exponent(y)),
                'memory_r2': float(hurst_r2(y))}
    except ImportError:
        return {'memory_hurst': np.nan, 'memory_r2': np.nan}
