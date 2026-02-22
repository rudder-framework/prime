"""Engine: embedding_dim — standalone false nearest neighbors."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 50:
        return {'embedding_dim_value': np.nan}
    try:
        from pmtvs import embedding_dimension
        return {'embedding_dim_value': float(embedding_dimension(y))}
    except ImportError:
        return {'embedding_dim_value': np.nan}
