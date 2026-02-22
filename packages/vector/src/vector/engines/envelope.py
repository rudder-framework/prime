"""Engine: envelope — RMS, peak, kurtosis of signal envelope."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    if n < 16:
        return {'envelope_rms': np.nan, 'envelope_peak': np.nan,
                'envelope_kurtosis': np.nan}
    try:
        from scipy.signal import hilbert
        analytic = hilbert(y)
        env = np.abs(analytic)
    except ImportError:
        # Approximate envelope via rolling max of abs
        w = max(n // 20, 3)
        env = np.array([np.max(np.abs(y[max(0, i - w):i + w + 1]))
                        for i in range(n)])

    rms = float(np.sqrt(np.mean(env ** 2)))
    peak = float(np.max(env))
    m = np.mean(env)
    s = np.std(env, ddof=1)
    kurt = float(np.mean(((env - m) / s) ** 4) - 3.0) if s > 1e-15 else 0.0

    return {'envelope_rms': rms, 'envelope_peak': peak, 'envelope_kurtosis': kurt}
