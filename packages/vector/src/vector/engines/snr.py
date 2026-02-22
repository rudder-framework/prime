"""Engine: snr — signal-to-noise ratio."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'snr_db': np.nan, 'snr_linear': np.nan,
           'snr_signal_power': np.nan, 'snr_noise_power': np.nan}
    if len(y) < 10:
        return nan

    try:
        from pmtvs import signal_to_noise
        result = signal_to_noise(y)
        if not isinstance(result, dict):
            return nan
        return {
            'snr_db': float(result.get('db', np.nan)),
            'snr_linear': float(result.get('linear', np.nan)),
            'snr_signal_power': float(result.get('signal_power', np.nan)),
            'snr_noise_power': float(result.get('noise_power', np.nan)),
        }
    except (ImportError, Exception):
        return nan
