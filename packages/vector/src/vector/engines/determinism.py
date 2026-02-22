"""Engine: determinism — standalone determinism score."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    if len(y) < 50:
        return {'determinism_score': np.nan}
    try:
        from pmtvs import determinism
        return {'determinism_score': float(determinism(y))}
    except ImportError:
        return {'determinism_score': np.nan}
