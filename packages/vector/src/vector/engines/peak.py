"""Engine: peak — peak value and peak-to-peak."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {'peak_value': np.nan, 'peak_to_peak': np.nan}
    return {
        'peak_value': float(np.max(np.abs(y))),
        'peak_to_peak': float(np.max(y) - np.min(y)),
    }
