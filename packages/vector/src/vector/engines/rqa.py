"""Engine: rqa — recurrence rate, determinism, correlation dim, embedding dim."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'rqa_recurrence_rate': np.nan, 'rqa_determinism': np.nan,
           'rqa_correlation_dimension': np.nan, 'rqa_embedding_dim': np.nan}
    if n < 50:
        return nan

    try:
        from pmtvs import recurrence_rate, determinism, correlation_dimension
        rr = float(recurrence_rate(y))
        det = float(determinism(y))
        cd = float(correlation_dimension(y))
    except ImportError:
        rr = _recurrence_rate(y)
        det = np.nan
        cd = np.nan

    try:
        from pmtvs import embedding_dimension
        ed = float(embedding_dimension(y))
    except (ImportError, AttributeError):
        ed = np.nan

    return {
        'rqa_recurrence_rate': rr,
        'rqa_determinism': det,
        'rqa_correlation_dimension': cd,
        'rqa_embedding_dim': ed,
    }


def _recurrence_rate(y, threshold=None):
    """Simple recurrence rate estimate."""
    n = min(len(y), 500)
    y_sub = y[:n]
    if threshold is None:
        threshold = 0.1 * np.std(y_sub)
    dist = np.abs(y_sub[:, None] - y_sub[None, :])
    return float(np.sum(dist < threshold) / (n * n))
