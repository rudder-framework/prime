"""Engine: wavelet_stability — multi-scale energy distribution."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    keys = ['wavelet_energy_low', 'wavelet_energy_mid', 'wavelet_energy_high',
            'wavelet_energy_ratio', 'wavelet_entropy', 'wavelet_concentration',
            'wavelet_dominant_scale', 'wavelet_energy_drift',
            'wavelet_temporal_std', 'wavelet_intermittency']
    nan = {k: np.nan for k in keys}
    n = len(y)
    if n < 32:
        return nan
    # Approximate wavelet decomposition via band-pass filtering (FFT)
    fft = np.fft.rfft(y - np.mean(y))
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)
    total = np.sum(psd)
    if total < 1e-30:
        return nan
    low = np.sum(psd[freqs < 0.1])
    mid = np.sum(psd[(freqs >= 0.1) & (freqs < 0.3)])
    high = np.sum(psd[freqs >= 0.3])
    bands = np.array([low, mid, high])
    p = bands / total
    entropy = float(-np.sum(p * np.log(p + 1e-30))) / np.log(3)
    dom_scale = int(np.argmax(bands))
    ratio = float(high / low) if low > 1e-15 else np.nan
    # Temporal variation via windowed energy
    w = max(n // 8, 4)
    energies = [np.sum(y[i:i+w] ** 2) for i in range(0, n - w, w)]
    e_arr = np.array(energies) if energies else np.array([0.0])
    return {
        'wavelet_energy_low': float(low / total),
        'wavelet_energy_mid': float(mid / total),
        'wavelet_energy_high': float(high / total),
        'wavelet_energy_ratio': ratio,
        'wavelet_entropy': entropy,
        'wavelet_concentration': float(np.max(p)),
        'wavelet_dominant_scale': float(dom_scale),
        'wavelet_energy_drift': float(np.polyfit(np.arange(len(e_arr)), e_arr, 1)[0]) if len(e_arr) > 1 else 0.0,
        'wavelet_temporal_std': float(np.std(e_arr)),
        'wavelet_intermittency': float(np.std(e_arr) / np.mean(e_arr)) if np.mean(e_arr) > 1e-15 else 0.0,
    }
