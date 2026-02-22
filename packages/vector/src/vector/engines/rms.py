"""Engine: rms — root mean square."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {'rms_value': np.nan}
    return {'rms_value': float(np.sqrt(np.mean(y ** 2)))}
