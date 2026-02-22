"""Engine: thd — Total Harmonic Distortion."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'thd_percent': np.nan, 'thd_db': np.nan,
           'thd_fundamental_power': np.nan, 'thd_harmonic_power': np.nan,
           'thd_n_harmonics': np.nan}
    n = len(y)
    if n < 32:
        return nan
    fft = np.fft.rfft(y - np.mean(y))
    mag = np.abs(fft[1:])
    if len(mag) < 4 or np.max(mag) < 1e-15:
        return nan
    fund_idx = int(np.argmax(mag))
    fund_power = float(mag[fund_idx] ** 2)
    harm_power = 0.0
    n_harm = 0
    for h in range(2, 8):
        idx = h * (fund_idx + 1) - 1
        if idx < len(mag):
            harm_power += float(mag[idx] ** 2)
            n_harm += 1
    thd_ratio = np.sqrt(harm_power / fund_power) if fund_power > 1e-15 else 0.0
    thd_db = float(20 * np.log10(thd_ratio + 1e-30))
    return {
        'thd_percent': float(thd_ratio * 100),
        'thd_db': thd_db,
        'thd_fundamental_power': fund_power,
        'thd_harmonic_power': harm_power,
        'thd_n_harmonics': float(n_harm),
    }
