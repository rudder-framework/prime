"""Engine: fundamental_freq — standalone fundamental frequency analysis."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'fundamental_freq_value': np.nan, 'fundamental_freq_power': np.nan,
           'fundamental_freq_ratio': np.nan, 'fundamental_freq_confidence': np.nan}
    if n < 16:
        return nan
    fft = np.fft.rfft(y - np.mean(y))
    psd = np.abs(fft[1:]) ** 2
    if len(psd) == 0 or np.sum(psd) < 1e-30:
        return nan
    peak = int(np.argmax(psd))
    total = np.sum(psd)
    freq = float((peak + 1) / n)
    power = float(psd[peak])
    ratio = float(power / total)
    # Confidence: peak sharpness
    neighbors = psd[max(0, peak - 2):peak + 3]
    conf = float(power / np.mean(neighbors)) if np.mean(neighbors) > 0 else 0.0
    return {
        'fundamental_freq_value': freq,
        'fundamental_freq_power': power,
        'fundamental_freq_ratio': ratio,
        'fundamental_freq_confidence': conf,
    }
