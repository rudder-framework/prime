"""Engine: lyapunov — exponent, embedding dim, tau, confidence."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'lyapunov_exponent': np.nan, 'lyapunov_embedding_dim': np.nan,
           'lyapunov_embedding_tau': np.nan, 'lyapunov_confidence': np.nan}
    if n < 100:
        return nan

    try:
        from pmtvs import lyapunov_rosenstein
        result = lyapunov_rosenstein(y)
        if isinstance(result, (int, float)):
            return {'lyapunov_exponent': float(result),
                    'lyapunov_embedding_dim': np.nan,
                    'lyapunov_embedding_tau': np.nan,
                    'lyapunov_confidence': np.nan}
        return {
            'lyapunov_exponent': float(result.get('lyapunov', result) if isinstance(result, dict) else result),
            'lyapunov_embedding_dim': float(result.get('embedding_dim', np.nan)) if isinstance(result, dict) else np.nan,
            'lyapunov_embedding_tau': float(result.get('embedding_tau', np.nan)) if isinstance(result, dict) else np.nan,
            'lyapunov_confidence': float(result.get('confidence', np.nan)) if isinstance(result, dict) else np.nan,
        }
    except ImportError:
        return nan
