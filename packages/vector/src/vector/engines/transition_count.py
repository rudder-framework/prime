"""Engine: transition_count — number of state transitions."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 2:
        return {'transition_count_value': 0.0}
    median = np.median(y)
    above = y > median
    transitions = int(np.sum(np.diff(above.astype(int)) != 0))
    return {'transition_count_value': float(transitions)}
