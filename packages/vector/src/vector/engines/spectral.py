"""Engine: spectral — slope, dominant freq, entropy, centroid, bandwidth."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'spectral_slope': np.nan, 'spectral_dominant_freq': np.nan,
           'spectral_entropy': np.nan, 'spectral_centroid': np.nan,
           'spectral_bandwidth': np.nan}
    if n < 16:
        return nan

    xc = y - np.mean(y)
    fft = np.fft.rfft(xc)
    psd = np.abs(fft[1:]) ** 2
    if len(psd) == 0 or np.sum(psd) < 1e-30:
        return nan

    freqs = np.arange(1, len(psd) + 1) / n

    # Dominant frequency
    peak_idx = int(np.argmax(psd))
    dom_freq = float(freqs[peak_idx])

    # Spectral slope (log-log)
    log_f = np.log(freqs + 1e-30)
    log_p = np.log(psd + 1e-30)
    slope = float(np.polyfit(log_f, log_p, 1)[0])

    # Spectral entropy
    p_norm = psd / np.sum(psd)
    entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-30)))

    # Spectral centroid
    centroid = float(np.sum(freqs * psd) / np.sum(psd))

    # Spectral bandwidth
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)))

    return {
        'spectral_slope': slope,
        'spectral_dominant_freq': dom_freq,
        'spectral_entropy': entropy,
        'spectral_centroid': centroid,
        'spectral_bandwidth': bandwidth,
    }
