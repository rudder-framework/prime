"""Engine: pulsation_index — flow pulsation analysis."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 4:
        return {'pulsation_index_value': np.nan, 'pulsation_index_flow_mean': np.nan,
                'pulsation_index_flow_range': np.nan}
    m = float(np.mean(y))
    r = float(np.max(y) - np.min(y))
    pi_val = r / abs(m) if abs(m) > 1e-15 else np.nan
    return {'pulsation_index_value': pi_val, 'pulsation_index_flow_mean': m,
            'pulsation_index_flow_range': r}
