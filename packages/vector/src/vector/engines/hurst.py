"""Engine: hurst — Hurst exponent and R/S R²."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 32:
        return {'hurst_exponent': np.nan, 'hurst_r2': np.nan}

    try:
        from pmtvs import hurst_exponent
        h = float(hurst_exponent(y))
    except ImportError:
        h = _hurst_rs(y)

    try:
        from pmtvs import hurst_r2 as _hurst_r2_func
        r2 = float(_hurst_r2_func(y))
    except (ImportError, AttributeError):
        r2 = np.nan

    return {'hurst_exponent': h, 'hurst_r2': r2}


def _hurst_rs(y):
    """Simple R/S Hurst estimate."""
    n = len(y)
    max_k = min(int(np.log2(n)), 10)
    if max_k < 2:
        return 0.5

    sizes = [int(2 ** k) for k in range(2, max_k + 1)]
    rs_values = []

    for size in sizes:
        n_blocks = n // size
        if n_blocks < 1:
            continue
        rs_block = []
        for i in range(n_blocks):
            block = y[i * size:(i + 1) * size]
            m = np.mean(block)
            cumdev = np.cumsum(block - m)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(block, ddof=1)
            if s > 1e-15:
                rs_block.append(r / s)
        if rs_block:
            rs_values.append((np.log(size), np.log(np.mean(rs_block))))

    if len(rs_values) < 2:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y_rs = np.array([v[1] for v in rs_values])
    slope = np.polyfit(x, y_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))
