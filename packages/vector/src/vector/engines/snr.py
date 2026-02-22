"""Engine: snr — signal-to-noise ratio."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'snr_db': np.nan, 'snr_linear': np.nan,
           'snr_signal_power': np.nan, 'snr_noise_power': np.nan}
    n = len(y)
    if n < 10:
        return nan
    # Estimate signal as low-pass filtered version
    k = max(n // 20, 3)
    kernel = np.ones(k) / k
    signal_est = np.convolve(y, kernel, mode='same')
    noise = y - signal_est
    sig_power = float(np.mean(signal_est ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-30:
        return {'snr_db': 100.0, 'snr_linear': 1e10,
                'snr_signal_power': sig_power, 'snr_noise_power': noise_power}
    linear = sig_power / noise_power
    db = float(10 * np.log10(linear))
    return {'snr_db': db, 'snr_linear': linear,
            'snr_signal_power': sig_power, 'snr_noise_power': noise_power}
