"""Engine: frequency_bands — low/mid/high band powers."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {k: np.nan for k in [
        'frequency_bands_low', 'frequency_bands_low_rel',
        'frequency_bands_mid', 'frequency_bands_mid_rel',
        'frequency_bands_high', 'frequency_bands_high_rel',
        'frequency_bands_total_power', 'frequency_bands_nyquist']}
    if n < 16:
        return nan

    fft = np.fft.rfft(y - np.mean(y))
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)
    total = float(np.sum(psd))
    if total < 1e-30:
        return nan

    # Bands: low [0, 0.1), mid [0.1, 0.3), high [0.3, 0.5]
    low = float(np.sum(psd[freqs < 0.1]))
    mid = float(np.sum(psd[(freqs >= 0.1) & (freqs < 0.3)]))
    high = float(np.sum(psd[freqs >= 0.3]))

    return {
        'frequency_bands_low': low,
        'frequency_bands_low_rel': low / total,
        'frequency_bands_mid': mid,
        'frequency_bands_mid_rel': mid / total,
        'frequency_bands_high': high,
        'frequency_bands_high_rel': high / total,
        'frequency_bands_total_power': total,
        'frequency_bands_nyquist': 0.5,
    }
