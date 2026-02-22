"""Engine: attractor — embedding dimension, correlation dimension, delay."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'attractor_embedding_dim': np.nan, 'attractor_correlation_dim': np.nan,
           'attractor_delay': np.nan}
    if len(y) < 100:
        return nan
    try:
        from pmtvs import embedding_dimension, correlation_dimension, mutual_information_delay
        ed = float(embedding_dimension(y))
        cd = float(correlation_dimension(y))
        tau = float(mutual_information_delay(y))
        return {'attractor_embedding_dim': ed, 'attractor_correlation_dim': cd,
                'attractor_delay': tau}
    except (ImportError, AttributeError):
        return nan
