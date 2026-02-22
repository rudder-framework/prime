"""Engine: variance_ratio — ratio, 4-lag, 8-lag, stat."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 16:
        return {'variance_ratio_value': np.nan, 'variance_ratio_4': np.nan,
                'variance_ratio_8': np.nan, 'variance_ratio_stat': np.nan}

    def _vr(series, lag):
        diffs = np.diff(series)
        var1 = np.var(diffs)
        if var1 < 1e-15:
            return 1.0
        k_diffs = series[lag:] - series[:-lag]
        var_k = np.var(k_diffs)
        return float(var_k / (lag * var1))

    vr2 = _vr(y, 2)
    vr4 = _vr(y, 4) if n >= 20 else np.nan
    vr8 = _vr(y, 8) if n >= 32 else np.nan
    vr_stat = float((vr2 - 1) * np.sqrt(n)) if np.isfinite(vr2) else np.nan

    return {
        'variance_ratio_value': vr2,
        'variance_ratio_4': vr4,
        'variance_ratio_8': vr8,
        'variance_ratio_stat': vr_stat,
    }
