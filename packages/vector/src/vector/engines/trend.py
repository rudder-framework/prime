"""Engine: trend — slope, R², detrend std, CUSUM range."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 4:
        return {'trend_slope': np.nan, 'trend_r2': np.nan,
                'trend_detrend_std': np.nan, 'trend_cusum_range': np.nan}

    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    fitted = np.polyval(coeffs, x)
    residuals = y - fitted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    detrend_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    cusum = np.cumsum(y - np.mean(y))
    cusum_range = float(np.max(cusum) - np.min(cusum))

    return {
        'trend_slope': slope,
        'trend_r2': r2,
        'trend_detrend_std': detrend_std,
        'trend_cusum_range': cusum_range,
    }
