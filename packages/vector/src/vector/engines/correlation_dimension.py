"""Engine: correlation_dimension — standalone Grassberger-Procaccia."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 50:
        return {'correlation_dimension_value': np.nan}
    try:
        from pmtvs import correlation_dimension
        return {'correlation_dimension_value': float(correlation_dimension(y))}
    except ImportError:
        return {'correlation_dimension_value': np.nan}
