"""Engine: harmonics — fundamental frequency, amplitudes, THD."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {k: np.nan for k in [
        'harmonics_fundamental_freq', 'harmonics_fundamental_amplitude',
        'harmonics_2x', 'harmonics_3x', 'harmonics_thd']}
    if n < 32:
        return nan

    fft = np.fft.rfft(y - np.mean(y))
    mag = np.abs(fft[1:])
    if len(mag) < 4 or np.max(mag) < 1e-15:
        return nan

    fund_idx = int(np.argmax(mag))
    fund_freq = float((fund_idx + 1) / n)
    fund_amp = float(mag[fund_idx])

    h2_idx = 2 * (fund_idx + 1) - 1
    h3_idx = 3 * (fund_idx + 1) - 1
    h2 = float(mag[h2_idx]) if h2_idx < len(mag) else 0.0
    h3 = float(mag[h3_idx]) if h3_idx < len(mag) else 0.0

    harm_power = h2 ** 2 + h3 ** 2
    thd = float(np.sqrt(harm_power) / fund_amp) if fund_amp > 1e-15 else 0.0

    return {
        'harmonics_fundamental_freq': fund_freq,
        'harmonics_fundamental_amplitude': fund_amp,
        'harmonics_2x': h2,
        'harmonics_3x': h3,
        'harmonics_thd': thd,
    }
